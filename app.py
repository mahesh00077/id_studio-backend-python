# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import uuid
import base64
import io as iomodule
import re
import requests
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageOps
from datetime import datetime

# ---------------------------------------------------------------------------
# Tesseract OCR configuration
# ---------------------------------------------------------------------------
# pytesseract is only a Python wrapper. The actual Tesseract executable
# (tesseract.exe) must also be found. We automatically check:
#   1. TESSERACT_CMD environment variable
#   2. tesseract.exe in PATH
#   3. Common Windows installation locations
# If your installation is somewhere else, set TESSERACT_CMD to the full
# path of tesseract.exe.
import shutil
from pathlib import Path

try:
    import pytesseract

    def find_tesseract_executable():
        candidates = []

        # User can override everything with an environment variable.
        env_path = os.environ.get("TESSERACT_CMD")
        if env_path:
            candidates.append(env_path)

        # PATH lookup.
        path_result = shutil.which("tesseract")
        if path_result:
            candidates.append(path_result)

        # Common Windows installation locations.
        candidates.extend([
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
        ])

        # First existing executable wins.
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return os.path.abspath(candidate)

        return None

    TESSERACT_CMD = find_tesseract_executable()

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        TESSERACT_AVAILABLE = True
    else:
        TESSERACT_AVAILABLE = False

except ImportError:
    pytesseract = None
    TESSERACT_CMD = None
    TESSERACT_AVAILABLE = False

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    remove = None

app = FastAPI(title="School ID Studio - Image Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5000", "http://127.0.0.1:5000", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
DESIGN_DIR = os.path.join(UPLOAD_DIR, "designs")
STUDENT_PHOTO_DIR = os.path.join(UPLOAD_DIR, "student_photos")
OUTPUT_DIR = os.path.join(UPLOAD_DIR, "output")
for directory in [UPLOAD_DIR, DESIGN_DIR, STUDENT_PHOTO_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# IMPORTANT: matching is intentionally strict. Do not add generic words like
# 'student' or 'admission' by themselves because they create false positives.
FIELD_DEFINITIONS = {
    "studentName": {
        "label": "Student Name",
        "required": True,
        "keywords": [
            "student name", "student's name", "student s name", "name of student",
            "full name", "student full name", "name of the student", "name"
        ],
        "default_position": {"x": 20, "y": 30},
    },
    "fatherName": {
        "label": "Father's Name",
        "required": True,
        "keywords": [
            "father name", "father's name", "father s name", "fathers name",
            "name of father", "father's full name", "father name"
        ],
        "default_position": {"x": 20, "y": 42},
    },
    "motherName": {
        "label": "Mother's Name",
        "required": False,
        "keywords": [
            "mother name", "mother's name", "mother s name", "mothers name",
            "name of mother"
        ],
        "default_position": {"x": 20, "y": 54},
    },
    "class": {
        "label": "Class",
        "required": True,
        "keywords": ["class", "class/grade", "grade", "standard", "std"],
        "default_position": {"x": 20, "y": 66},
    },
    "section": {
        "label": "Section",
        "required": False,
        "keywords": ["section", "section/division", "division", "sec"],
        "default_position": {"x": 55, "y": 66},
    },
    "rollNumber": {
        "label": "Roll No",
        "required": False,
        "keywords": [
            "roll no", "roll no.", "roll number", "roll #", "roll#", "roll"
        ],
        "default_position": {"x": 20, "y": 78},
    },
    "admissionNumber": {
        "label": "Admission No",
        "required": False,
        "keywords": [
            "admission no", "admission no.", "admission number", "admission #",
            "admission#", "admission id", "admission id no", "admn no", "adm no"
        ],
        "default_position": {"x": 55, "y": 30},
    },
    "dob": {
        "label": "Date of Birth",
        "required": False,
        "keywords": [
            "date of birth", "date-of-birth", "dob", "d.o.b", "birth date", "birthdate"
        ],
        "default_position": {"x": 20, "y": 90},
    },
    "phone": {
        "label": "Mobile No",
        "required": False,
        "keywords": [
            "mobile no", "mobile no.", "mobile number", "phone no", "phone no.",
            "phone number", "contact no", "contact number", "mobile", "phone", "contact"
        ],
        "default_position": {"x": 20, "y": 102},
    },
    "address": {
        "label": "Address",
        "required": False,
        "keywords": ["address", "home address", "residential address", "permanent address"],
        "default_position": {"x": 20, "y": 114},
    },
}


def verify_token(authorization: str = Header(None)):
    return {"user_id": "dev_user" if not authorization else "authenticated_user"}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9#./' -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_regex(keyword: str) -> re.Pattern:
    k = normalize_text(keyword)
    escaped = re.escape(k)
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])", re.IGNORECASE)


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    # Upscaling helps small ID-card labels.
    max_side = max(img.size)
    if max_side < 1800:
        scale = min(3.0, 1800 / max_side)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    return gray


def run_ocr(image: Image.Image) -> List[str]:
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract")

    try:
        # PSM 11 is useful for scattered labels on ID-card designs.
        processed = preprocess_for_ocr(image)
        configs = ["--oem 3 --psm 11", "--oem 3 --psm 6"]
        lines = []
        for config in configs:
            text = pytesseract.image_to_string(processed, config=config)
            lines.extend([normalize_text(x) for x in text.splitlines() if normalize_text(x)])
        # Preserve order while removing duplicates.
        unique = []
        seen = set()
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique.append(line)
        return unique
    except Exception as exc:
        raise RuntimeError(f"OCR failed: {exc}")


def match_fields_with_ocr(ocr_lines: List[str]) -> List[Dict[str, Any]]:
    detected = []
    all_text = " | ".join(ocr_lines)

    for field_id, info in FIELD_DEFINITIONS.items():
        matched_text = None
        matched_keyword = None
        # Prefer longer/more-specific phrases first.
        for keyword in sorted(info["keywords"], key=len, reverse=True):
            pattern = keyword_regex(keyword)
            for line in ocr_lines:
                if pattern.search(line):
                    matched_text = line
                    matched_keyword = keyword
                    break
            if matched_text:
                break

        # Avoid the generic 'name' keyword matching Father/Mother lines.
        if field_id == "studentName" and matched_keyword == "name":
            if any(k in all_text for k in ["father name", "father s name", "mother name", "mother s name"]):
                # Only accept generic Name when OCR has an actual standalone name label.
                if not any(re.search(r"(?<![a-z])name\s*[:.-]?(?![a-z])", line) for line in ocr_lines):
                    matched_text = None

        if matched_text:
            pos = info["default_position"]
            detected.append({
                "id": field_id,
                "field": field_id,
                "label": info["label"],
                "x": pos["x"],
                "y": pos["y"],
                "required": info["required"],
                "detected_text": matched_text,
                "matched_keyword": matched_keyword,
            })

    # Sort in a stable, practical form order.
    order = {k: i for i, k in enumerate(FIELD_DEFINITIONS.keys())}
    detected.sort(key=lambda x: order.get(x["id"], 999))
    return detected


def detect_fields_from_design(image_data: bytes):
    img = Image.open(iomodule.BytesIO(image_data))
    ocr_lines = run_ocr(img)
    fields = match_fields_with_ocr(ocr_lines)
    return fields, ocr_lines


async def process_photo_with_bg_removal(image_data: bytes) -> bytes:
    img = Image.open(iomodule.BytesIO(image_data)).convert("RGBA")
    if REMBG_AVAILABLE:
        try:
            img = remove(img)
        except Exception as exc:
            print(f"rembg failed, using fallback: {exc}")
            img = simple_background_removal(img)
    else:
        img = simple_background_removal(img)

    # Keep transparent output. The ID-card editor can place it over the design.
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)

    output = iomodule.BytesIO()
    img.save(output, format="PNG", optimize=True)
    return output.getvalue()


def simple_background_removal(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    data = []
    for r, g, b, a in img.getdata():
        if r > 220 and g > 220 and b > 220:
            data.append((r, g, b, 0))
        else:
            data.append((r, g, b, a))
    img.putdata(data)
    return img


def load_image_from_url(url: str) -> bytes:
    if not url:
        raise ValueError("design_url is required")
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.content


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "python-backend",
        "ocr_available": TESSERACT_AVAILABLE,
        "rembg_available": REMBG_AVAILABLE,
    }


@app.post("/api/process/design/analyze")
async def analyze_design(file: UploadFile = File(...), auth: dict = Depends(verify_token)):
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Invalid design image type")
    contents = await file.read()
    try:
        fields, ocr_lines = detect_fields_from_design(contents)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to analyze design: {exc}")

    file_id = str(uuid.uuid4())
    ext = (file.filename or "design.png").split(".")[-1].lower()
    if ext not in {"jpg", "jpeg", "png", "webp"}:
        ext = "png"
    original_path = os.path.join(DESIGN_DIR, f"{file_id}_original.{ext}")
    with open(original_path, "wb") as f:
        f.write(contents)

    return {
        "success": True,
        "file_id": file_id,
        "design_url": f"/uploads/designs/{file_id}_original.{ext}",
        "fields": fields,
        "total_fields": len(fields),
        "ocr_text": ocr_lines,
        "message": f"Detected {len(fields)} field(s) from design",
    }


@app.post("/api/process/design/analyze-url")
async def analyze_design_url(request: Dict[str, Any], auth: dict = Depends(verify_token)):
    """Analyze a design already stored by the main School ID backend."""
    url = request.get("design_url")
    try:
        contents = load_image_from_url(url)
        fields, ocr_lines = detect_fields_from_design(contents)
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Could not download design: {exc}")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to analyze design: {exc}")

    return {
        "success": True,
        "fields": fields,
        "total_fields": len(fields),
        "ocr_text": ocr_lines,
        "message": f"Detected {len(fields)} field(s) from design",
    }


@app.post("/api/remove-background")
async def remove_background(file: UploadFile = File(...), auth: dict = Depends(verify_token)):
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Invalid photo type")
    contents = await file.read()
    try:
        processed = await process_photo_with_bg_removal(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {exc}")
    b64 = base64.b64encode(processed).decode("utf-8")
    return {
        "success": True,
        "processed_base64": f"data:image/png;base64,{b64}",
        "message": "Background removed successfully",
    }


@app.post("/api/id-card/generate")
async def generate_id_card(request: Dict[str, Any], auth: dict = Depends(verify_token)):
    """Standalone generator. Your main School Admin backend can still own credits."""
    try:
        student_data = request.get("student_data") or request.get("studentData") or {}
        positions = request.get("positions") or {}
        font_sizes = request.get("font_sizes") or request.get("fontSizes") or {}
        photo_position = request.get("photo_position") or request.get("photoPosition") or {"x": 25, "y": 20}
        photo_size = request.get("photo_size") or request.get("photoSize") or 130
        photo_base64 = request.get("photo")
        design_url = request.get("design_url")

        dpi = 300
        width_px = round((55 / 25.4) * dpi)
        height_px = round((88 / 25.4) * dpi)
        canvas = Image.new("RGB", (width_px, height_px), "white")

        if design_url:
            design_bytes = load_image_from_url(design_url)
            design_img = Image.open(iomodule.BytesIO(design_bytes)).convert("RGB")
            design_img = ImageOps.fit(design_img, (width_px, height_px), method=Image.Resampling.LANCZOS)
            canvas.paste(design_img, (0, 0))

        if photo_base64 and "," in photo_base64:
            raw = base64.b64decode(photo_base64.split(",", 1)[1])
            photo = Image.open(iomodule.BytesIO(raw)).convert("RGBA")
            size = max(40, int(float(photo_size) * (width_px / 600)))
            photo.thumbnail((size, size), Image.Resampling.LANCZOS)
            px = int((float(photo_position.get("x", 25)) / 100) * width_px)
            py = int((float(photo_position.get("y", 20)) / 100) * height_px)
            canvas.paste(photo, (px, py), photo)

        draw = ImageDraw.Draw(canvas)
        for field_id, value in student_data.items():
            if value in (None, ""):
                continue
            pos = positions.get(field_id, {"x": 20, "y": 30})
            fs = int(float(font_sizes.get(field_id, 14)) * (width_px / 600))
            fs = max(8, fs)
            try:
                font = ImageFont.truetype("arial.ttf", fs)
            except Exception:
                font = ImageFont.load_default()
            x = int((float(pos.get("x", 20)) / 100) * width_px)
            y = int((float(pos.get("y", 30)) / 100) * height_px)
            draw.text((x, y), str(value), fill="#1a202c", font=font)

        file_id = str(uuid.uuid4())
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}.jpg")
        canvas.save(output_path, "JPEG", quality=96, subsampling=0, dpi=(dpi, dpi), optimize=True)
        with open(output_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "success": True,
            "file_id": file_id,
            "download_url": f"/uploads/output/{file_id}.jpg",
            "image_base64": f"data:image/jpeg;base64,{b64}",
            "dimensions": {"width": width_px, "height": height_px, "mm": "55x88", "dpi": dpi},
            "message": "ID card generated successfully",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/id-card/download")
async def download_id_card(request: Dict[str, Any], auth: dict = Depends(verify_token)):
    file_id = request.get("file_id")
    if not file_id:
        raise HTTPException(status_code=400, detail="file_id required")
    file_path = os.path.join(OUTPUT_DIR, f"{file_id}.jpg")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="image/jpeg", filename=f"id_card_{file_id}.jpg")


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")