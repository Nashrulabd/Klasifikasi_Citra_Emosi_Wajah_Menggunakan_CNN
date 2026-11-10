"""
Dashboard Streamlit - Klasifikasi Citra Emosi Wajah dengan CNN
Jalankan:  streamlit run app.py

Letakkan file ini sejajar dengan folder `output/` yang dihasilkan notebook
(klasifikasi_emosi_perbaikan.ipynb).
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------
_base = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = _base / "output" if (_base / "output").is_dir() else _base

# Kompatibel dengan Streamlit lama (use_container_width) maupun baru (width="stretch")
_ver = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
         "neutral": "😐", "sad": "😢", "surprise": "😲"}
ID_NAME = {"angry": "Marah", "disgust": "Jijik", "fear": "Takut", "happy": "Senang",
           "neutral": "Netral", "sad": "Sedih", "surprise": "Terkejut"}

REQUIRED_FILES = {
    "dataset_summary.json": "ringkasan dataset",
    "class_distribution.csv": "distribusi kelas",
    "training_history.json": "riwayat pelatihan",
    "metrics.json": "metrik evaluasi",
    "classification_report.csv": "laporan per kelas",
    "confusion_matrix.csv": "confusion matrix",
    "class_names.json": "nama kelas",
    "final_facial_emotion_cnn.keras": "model CNN",
}


# --------------------------------------------------------------------------
# Pemuat data (di-cache; otomatis ter-refresh bila file berubah)
# --------------------------------------------------------------------------
def _stamp(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def read_json(path: str, stamp: float):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def read_csv(path: str, stamp: float, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


def load_json(out_dir: Path, name: str):
    p = out_dir / name
    return read_json(str(p), _stamp(p)) if p.exists() else None


def load_csv(out_dir: Path, name: str, **kw):
    p = out_dir / name
    return read_csv(str(p), _stamp(p), **kw) if p.exists() else None


@st.cache_resource(show_spinner="Memuat model CNN...")
def load_model(path: str, stamp: float):
    import tensorflow as tf  # impor di sini agar tab lain tetap cepat dibuka
    return tf.keras.models.load_model(path)


def find_model_path(out_dir: Path):
    for name in ("final_facial_emotion_cnn.keras", "facial_emotion_cnn_best.keras"):
        if (out_dir / name).exists():
            return out_dir / name
    return None


def show_image_if_exists(out_dir: Path, name: str, caption: str):
    p = out_dir / name
    if p.exists():
        st.image(str(p), caption=caption, **STRETCH)
    else:
        st.caption(f"({name} belum ada di folder output)")


# --------------------------------------------------------------------------
# Pra-pemrosesan gambar baru (HARUS sama dengan notebook)
#   - grayscale 48x48, nilai piksel 0-255
#   - JANGAN dibagi 255: normalisasi sudah dilakukan layer Rescaling di dalam model
#   - wajah dicari dengan beberapa detektor Haar dan beberapa sudut putar, sehingga
#     wajah yang miring/menoleh tetap bisa ditemukan lalu "diluruskan" sebelum di-crop
# --------------------------------------------------------------------------
CASCADE_FILES = ("haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml",
                 "haarcascade_frontalface_alt.xml")
TRY_ANGLES = (0, -15, 15, -30, 30, -45, 45)     # derajat; dicoba berurutan dari yang paling kecil
DETECT_MAX_SIDE = 512                            # gambar besar diperkecil dulu agar deteksi cepat
IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)


@st.cache_resource
def get_face_detectors():
    detectors = []
    try:
        for name in CASCADE_FILES:
            det = cv2.CascadeClassifier(cv2.data.haarcascades + name)
            if not det.empty():
                detectors.append(det)
    except Exception:
        pass
    return detectors


def open_image(file) -> Image.Image:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)                      # perbaiki rotasi foto HP
    if img.mode in ("RGBA", "LA", "P"):                     # PNG transparan -> latar putih
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))
    return img


def rotate_bound(gray: np.ndarray, angle: float):
    """Putar gambar (angle > 0 = berlawanan arah jarum jam) tanpa memotong sudutnya."""
    if angle == 0:
        return gray, IDENTITY
    h, w = gray.shape
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    rot = cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=int(gray.mean()))
    return rot, M


def to_original_polygon(M, rect, scale):
    """Ubah persegi (x, y, w, h) pada gambar yang sudah diputar menjadi poligon di gambar asli."""
    x, y, w, h = rect
    pts = np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=np.float32)
    inv = cv2.invertAffineTransform(M)
    return (cv2.transform(pts, inv)[0] / scale).astype(np.int32)


def _center_in_gray(M, box):
    """Titik tengah kotak pada gambar yang sudah diputar -> koordinat gambar sebelum diputar."""
    x, y, w, h = box
    c = np.array([[[x + w / 2, y + h / 2]]], dtype=np.float32)
    return cv2.transform(c, cv2.invertAffineTransform(M))[0, 0]


@st.cache_data(show_spinner="Mencari wajah...", max_entries=32)
def detect_face(gray: np.ndarray):
    """
    Cari wajah termasuk yang miring. Return (sudut_putar, (x, y, w, h)) atau None;
    kotak berada pada gambar yang sudah diputar sebesar sudut tersebut.

    Deteksi dianggap benar bila minimal 2 detektor/sudut sepakat pada lokasi yang sama
    (deteksi palsu biasanya hanya muncul dari satu detektor).
    """
    detectors = get_face_detectors()
    if not detectors:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    min_size = max(24, int(0.08 * min(gray.shape)))

    for min_nb, angles in ((5, TRY_ANGLES), (3, TRY_ANGLES[:5])):      # ketat dulu, lalu dilonggarkan
        cands = []                                                     # (sudut, kotak, titik tengah)
        for angle in angles:
            rot, M = rotate_bound(gray, angle)
            eq = clahe.apply(rot)
            for det in detectors:
                for box in det.detectMultiScale(eq, scaleFactor=1.1, minNeighbors=min_nb,
                                                minSize=(min_size, min_size)):
                    box = tuple(int(v) for v in box)
                    cands.append((angle, box, _center_in_gray(M, box)))
            # berhenti lebih awal bila di sudut ini ada wajah yang disepakati >= 2 detektor
            here = [c for c in cands if c[0] == angle]
            if any(sum(np.hypot(*(c[2] - o[2])) < 0.5 * c[1][2] for o in here) >= 2 for c in here):
                break

        if not cands:
            continue
        # kelompokkan kandidat yang menunjuk wajah yang sama
        clusters = []
        for c in cands:
            for cl in clusters:
                if np.hypot(*(c[2] - cl[0][2])) < 0.5 * cl[0][1][2]:
                    cl.append(c)
                    break
            else:
                clusters.append([c])
        best = max(clusters, key=lambda cl: (len(cl), np.mean([c[1][2] for c in cl])))
        if len(best) >= 2 or min_nb == 3:
            votes = lambda c: sum(o[0] == c[0] for o in best)         # dukungan pada sudut yang sama
            angle, box, _ = max(best, key=lambda c: (votes(c), -abs(c[0])))
            return angle, box
    return None


def crop_with_margin(rot: np.ndarray, box, margin: float = 0.10):
    x, y, w, h = box
    m = int(margin * max(w, h))
    H, W = rot.shape
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, W), min(y + h + m, H)
    return rot[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def preprocess_face(pil_img: Image.Image, size=(48, 48), use_face_crop=True, manual=None):
    """
    Return: wajah 48x48 (uint8), pesan info, status (True = wajah ditemukan / crop manual),
            poligon area crop pada gambar asli.
    manual = {"cx", "cy", "size", "angle"} (0-1, 0-1, 0-1, derajat) untuk crop manual.
    """
    rgb = np.array(pil_img.convert("RGB"))
    gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = min(1.0, DETECT_MAX_SIDE / max(gray_full.shape))
    gray = (cv2.resize(gray_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1 else gray_full)
    H, W = gray.shape

    crop, rect, M, found = None, None, IDENTITY, False
    info = "Deteksi wajah dimatikan, memakai crop persegi di tengah gambar."

    if manual is not None:
        rot, M = rotate_bound(gray, manual["angle"])
        rh, rw = rot.shape
        s = max(8, int(manual["size"] * min(rh, rw)))
        x0 = int(np.clip(manual["cx"] * rw - s / 2, 0, max(rw - s, 0)))
        y0 = int(np.clip(manual["cy"] * rh - s / 2, 0, max(rh - s, 0)))
        crop = rot[y0:y0 + s, x0:x0 + s]
        rect, found = (x0, y0, crop.shape[1], crop.shape[0]), True
        info = "Crop manual dipakai."
    elif use_face_crop:
        if not get_face_detectors():
            info = "Detektor wajah tidak tersedia, memakai crop persegi di tengah gambar."
        else:
            info = "Wajah tidak terdeteksi, memakai crop persegi di tengah gambar (kotak oranye)."
            res = detect_face(gray)
            if res is not None:
                angle, box = res
                rot, M = rotate_bound(gray, angle)
                crop, rect = crop_with_margin(rot, box)
                found = True
                info = "Wajah terdeteksi dan di-crop otomatis."
                if angle:
                    info += f" Gambar diluruskan {abs(angle)}° karena wajah miring."

    if crop is None:                                         # crop persegi (tanpa meng-gepengkan gambar)
        M = IDENTITY
        s = min(H, W)
        y0, x0 = (H - s) // 2, (W - s) // 2
        crop, rect = gray[y0:y0 + s, x0:x0 + s], (x0, y0, s, s)

    face = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return face, info, found, to_original_polygon(M, rect, scale)


def predict_probs(model, face48: np.ndarray, tta: bool = True) -> np.ndarray:
    x = face48.astype("float32")[None, ..., None]            # 0-255, bentuk (1,48,48,1)
    if tta:                                                  # rata-rata dengan versi cermin
        x = np.concatenate([x, x[:, :, ::-1, :]], axis=0)
    return model.predict(x, verbose=0).mean(axis=0)


def draw_polygon(pil_img: Image.Image, poly, found: bool = True):
    arr = np.array(pil_img).copy()
    color = (80, 220, 120) if found else (255, 165, 0)       # hijau = wajah, oranye = crop cadangan
    t = max(2, arr.shape[1] // 200)
    cv2.polylines(arr, [poly.reshape(-1, 1, 2)], True, color, t)
    return arr


# --------------------------------------------------------------------------
# Tab-tab dashboard
# --------------------------------------------------------------------------
def tab_dataset(out_dir: Path):
    st.header("Ringkasan Dataset")
    summary = load_json(out_dir, "dataset_summary.json")
    dist = load_csv(out_dir, "class_distribution.csv")
    if summary is None or dist is None:
        st.warning("File `dataset_summary.json` / `class_distribution.csv` belum ada. Jalankan notebook sampai Langkah 3.")
        return

    total, n_train, n_val = summary["total_images"], summary["train_images"], summary["val_images"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah kelas", summary["num_classes"])
    c2.metric("Total gambar", f"{total:,}")
    c3.metric("Data latih", f"{n_train:,}", f"{n_train / total:.0%}", delta_color="off")
    c4.metric("Data validasi", f"{n_val:,}", f"{n_val / total:.0%}", delta_color="off")
    st.caption(f"Ukuran gambar {summary['image_size'][0]}×{summary['image_size'][1]} piksel, "
               f"mode {summary['color_mode']}, pembagian data {summary['split']}.")

    st.subheader("Distribusi gambar per kelas")
    left, right = st.columns([3, 2])
    with left:
        st.bar_chart(dist.set_index("Emotion")["Total Images"])
    with right:
        d = dist.copy()
        d["Proporsi"] = (d["Total Images"] / d["Total Images"].sum() * 100).round(1).astype(str) + " %"
        st.dataframe(d, hide_index=True, **STRETCH)

    ratio = dist["Total Images"].max() / dist["Total Images"].min()
    big = dist.loc[dist["Total Images"].idxmax(), "Emotion"]
    small = dist.loc[dist["Total Images"].idxmin(), "Emotion"]
    if ratio > 1.5:
        st.info(f"Terdapat **class imbalance**: kelas *{big}* {ratio:.1f}× lebih banyak dibanding *{small}*. "
                "Hal ini ditangani dengan `class_weight` saat pelatihan.")
    else:
        st.success("Distribusi kelas relatif seimbang.")

    with st.expander("Contoh citra tiap kelas"):
        show_image_if_exists(out_dir, "2_sample_images.png", "Contoh citra ekspresi wajah")


def tab_training(out_dir: Path):
    st.header("Hasil Pelatihan")
    hist = load_json(out_dir, "training_history.json")
    if hist is None:
        st.warning("File `training_history.json` belum ada. Jalankan notebook sampai bagian pelatihan.")
        return

    epochs = len(hist["loss"])
    best = int(np.argmin(hist["val_loss"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Epoch dijalankan", epochs)
    c2.metric("Epoch terbaik", best + 1)
    c3.metric("Val accuracy terbaik", f"{max(hist['val_accuracy']):.2%}")
    c4.metric("Val loss terendah", f"{min(hist['val_loss']):.4f}")

    idx = pd.Index(range(1, epochs + 1), name="Epoch")
    left, right = st.columns(2)
    with left:
        st.subheader("Akurasi")
        st.line_chart(pd.DataFrame({"Training": hist["accuracy"], "Validasi": hist["val_accuracy"]}, index=idx))
    with right:
        st.subheader("Loss")
        st.line_chart(pd.DataFrame({"Training": hist["loss"], "Validasi": hist["val_loss"]}, index=idx))

    gap = hist["accuracy"][best] - hist["val_accuracy"][best]
    if gap > 0.15:
        st.warning(f"Selisih akurasi training–validasi pada epoch terbaik {gap:.1%}: indikasi **overfitting**.")
    else:
        st.success(f"Selisih akurasi training–validasi pada epoch terbaik {gap:.1%}: model cukup stabil.")

    metrics = load_json(out_dir, "metrics.json") or {}
    if "training_time_sec" in metrics:
        st.caption(f"Waktu pelatihan: {metrics['training_time_sec'] / 60:.1f} menit. "
                   "Learning rate diturunkan otomatis (ReduceLROnPlateau) dan pelatihan dihentikan dini (EarlyStopping).")

    with st.expander("Grafik dari notebook"):
        show_image_if_exists(out_dir, "3_training_curves.png", "Kurva akurasi dan loss")


def tab_evaluation(out_dir: Path):
    st.header("Evaluasi Model")
    metrics = load_json(out_dir, "metrics.json")
    report = load_csv(out_dir, "classification_report.csv", index_col=0)
    cm_df = load_csv(out_dir, "confusion_matrix.csv", index_col=0)
    if metrics is None:
        st.warning("File `metrics.json` belum ada. Jalankan notebook sampai Langkah 5.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall", f"{metrics['recall']:.2%}")
    c4.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    extra = "Precision, Recall, dan F1-Score memakai rata-rata tertimbang (weighted) seluruh kelas."
    if "macro_f1" in metrics:
        extra += f" Macro F1 (semua kelas dianggap sama penting): {metrics['macro_f1']:.2%}."
    st.caption(extra)

    if report is not None:
        st.subheader("Laporan per kelas")
        class_rows = [r for r in report.index if r not in ("accuracy", "macro avg", "weighted avg")]
        show = report.copy()
        show.index.name = "Kelas"
        fmt = {c: "{:.3f}" for c in show.columns if c != "support"}
        fmt["support"] = "{:.0f}"
        st.dataframe(show.style.format(fmt, na_rep="-"), **STRETCH)

        f1 = report.loc[class_rows, "f1-score"]
        st.bar_chart(f1.rename("F1-score"))
        st.markdown(
            f"- Kelas **paling mudah** dikenali: **{f1.idxmax()}** (F1 = {f1.max():.2f})\n"
            f"- Kelas **paling sulit** dikenali: **{f1.idxmin()}** (F1 = {f1.min():.2f})"
        )

    if cm_df is not None:
        st.subheader("Confusion Matrix")
        mode = st.radio("Tampilan", ["Jumlah gambar", "Proporsi per kelas asli"], horizontal=True)
        cm = cm_df.values.astype(float)
        if mode.startswith("Proporsi"):
            data, fmt_str = cm / cm.sum(axis=1, keepdims=True), ".2f"
        else:
            data, fmt_str = cm, ".0f"
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(data, annot=True, fmt=fmt_str, cmap="Blues",
                    xticklabels=cm_df.columns, yticklabels=cm_df.index, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label asli")
        st.pyplot(fig)
        plt.close(fig)

        off = cm.copy()
        np.fill_diagonal(off, 0)
        i, j = np.unravel_index(off.argmax(), off.shape)
        if off[i, j] > 0:
            st.caption(f"Kesalahan paling sering: gambar **{cm_df.index[i]}** diprediksi sebagai "
                       f"**{cm_df.columns[j]}** ({int(off[i, j])} gambar).")

    st.subheader("Area wajah yang memengaruhi keputusan model (Grad-CAM)")
    show_image_if_exists(out_dir, "gradcam_visualization.png", "Warna merah/kuning = area paling berpengaruh")
    with st.expander("Contoh prediksi pada data validasi"):
        show_image_if_exists(out_dir, "7_sample_predictions.png", "Hijau = benar, merah = salah")


def tab_predict(out_dir: Path):
    st.header("Uji Prediksi Gambar Baru")
    st.write("Unggah foto wajah (JPG/PNG/BMP/WEBP). Hasil prediksi muncul otomatis.")

    model_path = find_model_path(out_dir)
    names = load_json(out_dir, "class_names.json")
    if model_path is None or names is None:
        st.warning("Model (`final_facial_emotion_cnn.keras`) atau `class_names.json` belum ada di folder output.")
        return
    model = load_model(str(model_path), _stamp(model_path))
    size = tuple(model.input_shape[1:3])

    o1, o2, o3 = st.columns(3)
    use_crop = o1.checkbox("Deteksi & crop wajah otomatis (disarankan untuk foto penuh)", value=True)
    use_tta = o2.checkbox("Rata-ratakan dengan gambar cermin (hasil lebih stabil)", value=True)
    use_manual = o3.checkbox("Crop manual (jika wajah tidak terdeteksi)", value=False)

    uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Belum ada gambar. Gunakan foto wajah **menghadap depan, utuh, dan cukup terang** untuk hasil terbaik.")
        return

    try:
        img = open_image(uploaded)
    except Exception as e:
        st.error(f"Gambar tidak dapat dibaca: {e}")
        return

    manual = None
    if use_manual:
        s1, s2, s3, s4 = st.columns(4)
        cx = s1.slider("Posisi horizontal (%)", 0, 100, 50)
        cy = s2.slider("Posisi vertikal (%)", 0, 100, 50)
        sz = s3.slider("Ukuran kotak (%)", 10, 100, 60)
        ang = s4.slider("Putar gambar (°, + = berlawanan jarum jam)", -60, 60, 0)
        manual = {"cx": cx / 100, "cy": cy / 100, "size": sz / 100, "angle": ang}

    face48, info, found, poly = preprocess_face(img, size, use_crop, manual)
    probs = predict_probs(model, face48, use_tta)
    top = int(np.argmax(probs))
    label, conf = names[top], float(probs[top])

    col1, col2, col3 = st.columns([1, 1, 1.3])
    with col1:
        st.subheader("Gambar asli")
        st.image(draw_polygon(img, poly, found), **STRETCH)
        st.caption("Kotak = area yang dikirim ke model (hijau: wajah ditemukan, oranye: crop cadangan).")
    with col2:
        st.subheader(f"Input model ({size[0]}×{size[1]} grayscale)")
        st.image(face48, width=240, clamp=True)
        if found or not use_crop:
            st.caption(info)
        else:
            st.warning(info + " Centang **Crop manual** lalu geser slider sampai kotak menutupi wajah.")
    with col3:
        st.subheader("Hasil prediksi")
        st.success(f"{EMOJI.get(label, '🙂')} **{label.upper()}** ({ID_NAME.get(label, label)}), "
                   f"keyakinan {conf:.2%}")
        prob_df = pd.DataFrame({"Emosi": names, "Probabilitas (%)": probs * 100}).set_index("Emosi")
        st.bar_chart(prob_df)

    order = np.argsort(probs)[::-1][:3]
    st.markdown("**3 prediksi teratas:** " + " · ".join(
        f"{EMOJI.get(names[i], '')} {names[i]} {probs[i]:.1%}" for i in order))

    if conf < 0.5:
        tips = ["Pakai foto wajah **utuh, menghadap depan**, dan tidak terpotong.",
                "Pastikan pencahayaan cukup dan wajah tidak tertutup tangan/masker/kacamata gelap."]
        if not found and use_crop:
            tips.insert(0, "Wajah **tidak terdeteksi**, sehingga yang dipakai hanya potongan tengah gambar.")
        st.info("Keyakinan model masih rendah. Beberapa saran:\n\n" + "\n".join(f"- {t}" for t in tips) +
                "\n\nModel dilatih pada FER2013 (wajah kecil 48×48), sehingga foto yang sangat berbeda dari "
                "data latih memang lebih sulit diprediksi.")


# --------------------------------------------------------------------------
# Halaman utama
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Klasifikasi Emosi Wajah (CNN)", page_icon="🙂", layout="wide")

    with st.sidebar:
        st.header("⚙️ Pengaturan")
        out_dir = Path(st.text_input("Folder output notebook", str(DEFAULT_OUT_DIR)))
        st.markdown("**Status file**")
        for name, desc in REQUIRED_FILES.items():
            st.markdown(f"{'✅' if (out_dir / name).exists() else '❌'} {desc}")
        if st.button("🔄 Muat ulang data"):
            st.cache_data.clear()
            st.rerun()

    st.title("🙂 Klasifikasi Citra Emosi Wajah dengan CNN")
    st.markdown(
        "Dashboard hasil praktikum **Convolutional Neural Network (CNN)** untuk mengenali 7 emosi wajah "
        "(*angry, disgust, fear, happy, neutral, sad, surprise*) pada dataset **Facial Emotion Recognition "
        "(FER2013)** dari Kaggle. Citra diubah ke grayscale 48×48 piksel, dinormalisasi, lalu diklasifikasikan "
        "oleh CNN 4 blok konvolusi."
    )
    if not out_dir.exists():
        st.error(f"Folder `{out_dir}` tidak ditemukan. Jalankan notebook lebih dulu atau ubah path di sidebar.")
        st.stop()

    t1, t2, t3, t4 = st.tabs(["📊 Dataset", "📈 Pelatihan", "✅ Evaluasi", "🔮 Prediksi Gambar"])
    with t1:
        tab_dataset(out_dir)
    with t2:
        tab_training(out_dir)
    with t3:
        tab_evaluation(out_dir)
    with t4:
        tab_predict(out_dir)


if __name__ == "__main__":
    main()
