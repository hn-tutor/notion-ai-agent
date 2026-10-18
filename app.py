"""
Notion AI Agent — Streamlit app.
Nhận lịch trình từ văn bản hoặc ảnh, trích xuất sự kiện bằng Gemini, lưu vào Notion.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any
from zoneinfo import ZoneInfo

import google.generativeai as genai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from notion_client import Client
from notion_client.errors import APIResponseError
from PIL import Image

load_dotenv()

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
ALLOWED_CATEGORIES = ("Công việc", "Học tập", "Cá nhân")
GEMINI_MODEL = "gemini-2.5-flash"


def now_vietnam() -> datetime:
    return datetime.now(VN_TZ)


def iso_now() -> str:
    return now_vietnam().isoformat(timespec="seconds")


def load_env_defaults() -> dict[str, str]:
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "notion_token": os.getenv("notion_token", "").strip(),
        "notion_database_id": os.getenv("notion_database_id", "").strip(),
    }


def build_system_prompt(reference_now: datetime) -> str:
    weekday_vi = (
        "Thứ Hai",
        "Thứ Ba",
        "Thứ Tư",
        "Thứ Năm",
        "Thứ Sáu",
        "Thứ Bảy",
        "Chủ Nhật",
    )[reference_now.weekday()]
    now_iso = reference_now.isoformat(timespec="seconds")

    return f"""Bạn là trợ lý lịch trình. Nhiệm vụ: đọc văn bản và/hoặc hình ảnh (screenshot tin nhắn, thời khóa biểu, lịch họp) rồi trích xuất MỌI sự kiện thành JSON.

Thời điểm tham chiếu (múi giờ Việt Nam, UTC+7):
- Bây giờ: {now_iso}
- Ngày trong tuần hiện tại: {weekday_vi}
- Múi giờ bắt buộc: +07:00

Quy tắc thời gian:
- Mọi ngày tương đối phải được quy ra ngày tuyệt đối dựa trên thời điểm tham chiếu.
  Ví dụ: "hôm nay", "ngày mai", "thứ 6 tuần này", "tuần sau", "sáng mai".
- start_time và end_time (nếu có) PHẢI là ISO 8601 đầy đủ: YYYY-MM-DDTHH:MM:SS+07:00
- Nếu chỉ có ngày, không có giờ: dùng 09:00:00+07:00 cho start_time.
- Nếu có giờ bắt đầu nhưng không có giờ kết thúc: end_time = null.
- Nếu có khoảng thời gian (ví dụ 14:00-16:00): điền cả start_time và end_time.
- Không bịa sự kiện. Chỉ trích xuất những gì có trong nguồn.

Quy tắc category (chọn đúng 1 trong 3):
- "Công việc": họp, deadline, công ty, khách hàng, standup
- "Học tập": lớp học, bài tập, thi, thời khóa biểu, học nhóm
- "Cá nhân": cá nhân, gia đình, hẹn bạn, sức khỏe, việc nhà
- Nếu không rõ: "Cá nhân"

Quy tắc khác:
- title: ngắn gọn, rõ nghĩa (tiếng Việt nếu nguồn tiếng Việt).
- notes: chi tiết phụ, địa điểm, người tham gia, link. Chuỗi rỗng nếu không có.
- Trả về DUY NHẤT một JSON object, không markdown, không giải thích.

Schema bắt buộc:
{{
  "events": [
    {{
      "title": "string",
      "start_time": "YYYY-MM-DDTHH:MM:SS+07:00",
      "end_time": "YYYY-MM-DDTHH:MM:SS+07:00 hoặc null",
      "category": "Công việc | Học tập | Cá nhân",
      "notes": "string"
    }}
  ]
}}

Nếu không tìm thấy sự kiện nào: {{"events": []}}
"""


def extract_json_object(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Gemini trả về nội dung rỗng.")

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("Không parse được JSON từ phản hồi của Gemini.")


def normalize_iso(value: Any, *, allow_null: bool = False) -> str | None:
    if value is None or value == "" or str(value).lower() == "null":
        if allow_null:
            return None
        raise ValueError("Thiếu thời gian bắt đầu.")

    raw = str(value).strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Định dạng thời gian không hợp lệ: {raw}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=VN_TZ)
    else:
        parsed = parsed.astimezone(VN_TZ)
    return parsed.isoformat(timespec="seconds")


def normalize_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("events")
        if items is None:
            raise ValueError("JSON thiếu trường 'events'.")
    else:
        raise ValueError("JSON không đúng kiểu danh sách sự kiện.")

    if not isinstance(items, list):
        raise ValueError("'events' phải là một mảng.")

    cleaned: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Sự kiện #{index} không phải object.")

        title = str(item.get("title") or "").strip()
        if not title:
            raise ValueError(f"Sự kiện #{index} thiếu title.")

        start_time = normalize_iso(item.get("start_time"), allow_null=False)
        end_time = normalize_iso(item.get("end_time"), allow_null=True)

        if end_time and start_time and end_time < start_time:
            end_time = None

        category = str(item.get("category") or "Cá nhân").strip()
        if category not in ALLOWED_CATEGORIES:
            category = "Cá nhân"

        notes = str(item.get("notes") or "").strip()

        cleaned.append(
            {
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
                "category": category,
                "notes": notes,
            }
        )
    return cleaned


def analyze_schedule(
    *,
    gemini_api_key: str,
    text: str,
    image_bytes: bytes | None,
    image_mime: str | None,
) -> list[dict[str, Any]]:
    genai.configure(api_key=gemini_api_key)
    reference_now = now_vietnam()
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=build_system_prompt(reference_now),
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )

    parts: list[Any] = [
        (
            "Hãy trích xuất tất cả sự kiện từ nguồn dưới đây. "
            f"Thời điểm hiện tại là {reference_now.isoformat(timespec='seconds')}."
        )
    ]
    if text.strip():
        parts.append(f"Văn bản lịch trình:\n{text.strip()}")
    if image_bytes:
        image = Image.open(BytesIO(image_bytes))
        parts.append(image)
        parts.append(f"(Ảnh đính kèm, MIME: {image_mime or 'unknown'})")

    response = model.generate_content(parts)
    raw = getattr(response, "text", "") or ""
    payload = extract_json_object(raw)
    return normalize_events(payload)


def notion_date_payload(start_time: str, end_time: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"start": start_time}
    if end_time:
        payload["end"] = end_time
    return payload


def save_events_to_notion(
    *,
    notion_token: str,
    database_id: str,
    events: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    client = Client(auth=notion_token)
    errors: list[str] = []
    saved = 0

    for event in events:
        try:
            client.pages.create(
                parent={"database_id": database_id},
                properties={
                    "Name": {
                        "title": [{"type": "text", "text": {"content": event["title"][:2000]}}]
                    },
                    "Date": {
                        "date": notion_date_payload(event["start_time"], event["end_time"])
                    },
                    "Category": {"select": {"name": event["category"]}},
                    "Notes": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": (event["notes"] or "")[:2000]},
                            }
                        ]
                    },
                },
            )
            saved += 1
        except APIResponseError as exc:
            errors.append(f"{event['title']}: {exc}")
        except Exception as exc:  # noqa: BLE001 — hiển thị lỗi từng sự kiện trên UI
            errors.append(f"{event['title']}: {exc}")

    return saved, errors


def init_session_state() -> None:
    if "events" not in st.session_state:
        st.session_state.events = []
    if "analyze_error" not in st.session_state:
        st.session_state.analyze_error = ""


def render_sidebar(defaults: dict[str, str]) -> dict[str, str]:
    with st.sidebar:
        st.header("Cấu hình")
        st.caption(
            "Ưu tiên đọc từ file `.env`. Nếu chưa có, nhập trực tiếp tại đây."
        )
        gemini_api_key = st.text_input(
            "GEMINI_API_KEY",
            value=defaults["gemini_api_key"],
            type="password",
            help="API key từ Google AI Studio.",
        )
        notion_token = st.text_input(
            "notion_token",
            value=defaults["notion_token"],
            type="password",
            help="Internal Integration Token của Notion.",
        )
        notion_database_id = st.text_input(
            "notion_database_id",
            value=defaults["notion_database_id"],
            help="ID database Notion (cột Name, Date, Category, Notes).",
        )

        st.divider()
        st.markdown(
            f"""
**Thời điểm hệ thống:** `{iso_now()}`

**Cột Notion cần có:**
- `Name` (title)
- `Date` (date)
- `Category` (select: Công việc / Học tập / Cá nhân)
- `Notes` (rich_text)

Hãy share database cho Integration trước khi lưu.
"""
        )
    return {
        "gemini_api_key": gemini_api_key.strip(),
        "notion_token": notion_token.strip(),
        "notion_database_id": notion_database_id.strip(),
    }


def events_to_dataframe(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for event in events:
        rows.append(
            {
                "Tiêu đề": event["title"],
                "Bắt đầu": event["start_time"],
                "Kết thúc": event["end_time"] or "",
                "Danh mục": event["category"],
                "Ghi chú": event["notes"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="Notion AI Agent",
        page_icon="📅",
        layout="wide",
    )
    init_session_state()
    credentials = render_sidebar(load_env_defaults())

    st.title("📅 Notion AI Agent")
    st.write(
        "Dán lịch trình dạng chữ hoặc tải ảnh chụp màn hình. "
        "Gemini sẽ nhận diện sự kiện, bạn xem trước rồi lưu vào Notion Database."
    )

    col_text, col_image = st.columns(2)
    with col_text:
        schedule_text = st.text_area(
            "Lịch trình dạng chữ",
            height=220,
            placeholder="Ví dụ: Ngày mai họp khách hàng 14:00-15:30. Thứ 6 tuần này thi giữa kỳ 8h sáng.",
        )
    with col_image:
        uploaded = st.file_uploader(
            "Tải ảnh lịch trình (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            st.image(uploaded, caption="Ảnh đã tải lên", use_container_width=True)

    analyze_clicked = st.button("Phân tích lịch trình", type="primary", use_container_width=True)

    if analyze_clicked:
        image_bytes = uploaded.getvalue() if uploaded is not None else None
        image_mime = uploaded.type if uploaded is not None else None

        if not credentials["gemini_api_key"]:
            st.error("Thiếu GEMINI_API_KEY. Điền trên sidebar hoặc trong file .env.")
        elif not schedule_text.strip() and not image_bytes:
            st.warning("Hãy nhập văn bản hoặc tải lên một ảnh trước khi phân tích.")
        else:
            with st.spinner("Gemini đang đọc lịch trình..."):
                try:
                    events = analyze_schedule(
                        gemini_api_key=credentials["gemini_api_key"],
                        text=schedule_text,
                        image_bytes=image_bytes,
                        image_mime=image_mime,
                    )
                    st.session_state.events = events
                    st.session_state.analyze_error = ""
                    if events:
                        st.success(f"Nhận diện được {len(events)} sự kiện.")
                    else:
                        st.info("Không tìm thấy sự kiện nào trong nội dung đã gửi.")
                except Exception as exc:  # noqa: BLE001
                    st.session_state.events = []
                    st.session_state.analyze_error = str(exc)
                    st.error(f"Phân tích thất bại: {exc}")

    events: list[dict[str, Any]] = st.session_state.events
    if events:
        st.subheader("Xem trước sự kiện")
        st.dataframe(events_to_dataframe(events), use_container_width=True, hide_index=True)

        save_clicked = st.button("Lưu vào Notion", type="secondary", use_container_width=True)
        if save_clicked:
            if not credentials["notion_token"] or not credentials["notion_database_id"]:
                st.error("Thiếu notion_token hoặc notion_database_id.")
            else:
                with st.spinner("Đang ghi sự kiện vào Notion..."):
                    try:
                        saved, errors = save_events_to_notion(
                            notion_token=credentials["notion_token"],
                            database_id=credentials["notion_database_id"],
                            events=events,
                        )
                        if saved:
                            st.success(f"Đã lưu {saved}/{len(events)} sự kiện vào Notion.")
                        if errors:
                            st.error("Một số sự kiện lưu thất bại:")
                            for message in errors:
                                st.write(f"- {message}")
                        if saved == len(events) and not errors:
                            st.balloons()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Không kết nối được Notion: {exc}")


if __name__ == "__main__":
    main()
