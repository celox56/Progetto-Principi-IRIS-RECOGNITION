import os
import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import hamming
from itertools import combinations, product
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv  # <--- aggiunto

# ======================================================
# CONFIGURAZIONE
# ======================================================

DATASET_PATH = "/Users/michele/Desktop/Progetto Principi/Eye database"

RADIAL_RES = 64
ANGULAR_RES = 256

MAX_IMAGES_PER_PERSON = 3
MAX_IMPOSTOR_PAIRS = 5000

EXPORT_DIR = "exporto_slide"
CSV_DIR = "File CSV"   # <--- nuova cartella per i CSV


# ======================================================
# UTILS
# ======================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_plot_style():
    plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": 140,
        "figure.figsize": (7.5, 5.0),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
    })


# ======================================================
# 1. SEGMENTAZIONE IRIDE
# ======================================================

def segment_iris(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape[:2]

    circles_pupil = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=80,
        param2=15,
        minRadius=8,
        maxRadius=int(min(h, w) / 3)
    )

    if circles_pupil is not None:
        xp, yp, rp = np.uint16(np.around(circles_pupil))[0][0]
    else:
        xp, yp = w // 2, h // 2
        rp = int(0.15 * min(h, w))

    max_ri = int(0.9 * min(h, w) / 2)
    ri = int(min(max_ri, rp * 1.8))

    return (int(xp), int(yp), int(rp)), (int(xp), int(yp), int(ri))


# ======================================================
# 2. NORMALIZZAZIONE – RUBBER SHEET
# ======================================================

def normalize_iris(img, pupil, iris, radial_res=RADIAL_RES, angular_res=ANGULAR_RES):
    xp, yp, rp = pupil
    xi, yi, ri = iris

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    theta = np.linspace(0, 2 * np.pi, angular_res, endpoint=False)
    r = np.linspace(0, 1, radial_res)

    polar = np.zeros((radial_res, angular_res), dtype=np.float32)

    for i, ang in enumerate(theta):
        ct = np.cos(ang)
        st = np.sin(ang)
        for j, rr in enumerate(r):
            r_act = rp + rr * (ri - rp)

            x = xp + r_act * ct
            y = yp + r_act * st

            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)

            polar[j, i] = ndimage.map_coordinates(
                gray,
                [[y], [x]],
                order=1,
                mode="reflect"
            )

    return polar / 255.0


# ======================================================
# 3. GABOR-LIKE FEATURE EXTRACTION
# ======================================================

def gabor_filtering(polar, sigma=2.5):
    g_real = ndimage.gaussian_filter(polar, sigma=sigma)
    threshold = np.mean(g_real)
    iris_code = (g_real > threshold).astype(np.uint8)

    g_imag = ndimage.gaussian_laplace(polar, sigma=sigma)
    mask = (np.abs(g_imag) > 1e-4).astype(np.uint8)

    return iris_code.flatten(), mask.flatten()


# ======================================================
# 4. DISTANZA DI HAMMING
# ======================================================

def hamming_distance(code1, mask1, code2, mask2):
    valid = np.logical_and(mask1 == 1, mask2 == 1)
    if np.sum(valid) == 0:
        return 1.0
    return hamming(code1[valid], code2[valid])


# ======================================================
# 5. PROCESSING DI UNA SOLA IMMAGINE
# ======================================================

def process_iris(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Impossibile leggere l'immagine: {image_path}")
    pupil, iris = segment_iris(img)
    polar = normalize_iris(img, pupil, iris)
    code, mask = gabor_filtering(polar)
    return code, mask


# ======================================================
# 6. CARICAMENTO DATASET
# ======================================================

def load_dataset(dataset_path, max_images_per_person=MAX_IMAGES_PER_PERSON):
    people = {}

    def sort_key(name):
        return (0, int(name)) if name.isdigit() else (1, name)

    for person in sorted(os.listdir(dataset_path), key=sort_key):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue

        imgs = []
        for fname in sorted(os.listdir(person_dir)):
            low = fname.lower()
            if low.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                imgs.append(os.path.join(person_dir, fname))

        if imgs:
            people[person] = imgs[:max_images_per_person]

    return people


# ======================================================
# 7. CALCOLO IRISCODES
# ======================================================

def compute_all_codes(dataset_path):
    dataset = load_dataset(dataset_path)
    codes = {}

    for person, img_list in dataset.items():
        entries = []
        for img_path in img_list:
            try:
                code, mask = process_iris(img_path)
                entries.append((img_path, code, mask))
                print(f"[OK] {person} - {os.path.basename(img_path)}")
            except Exception as e:
                print(f"[SKIP] {person} - {os.path.basename(img_path)} -> {e}")
        if entries:
            codes[person] = entries

    return codes


# ======================================================
# 8. GENUINE & IMPOSTOR DISTANCES
# ======================================================

def compute_distances(codes_dict, max_impostor_pairs=MAX_IMPOSTOR_PAIRS):
    rng = np.random.default_rng(0)

    persons = list(codes_dict.keys())
    genuine = []
    impostor = []

    for person in persons:
        entries = codes_dict[person]
        if len(entries) < 2:
            continue
        for (_, c1, m1), (_, c2, m2) in combinations(entries, 2):
            genuine.append(hamming_distance(c1, m1, c2, m2))

    all_pairs = []
    for p1, p2 in combinations(persons, 2):
        entries1 = codes_dict[p1]
        entries2 = codes_dict[p2]
        for (_, c1, m1), (_, c2, m2) in product(entries1, entries2):
            all_pairs.append((c1, m1, c2, m2))

    if len(all_pairs) > max_impostor_pairs:
        idx = rng.choice(len(all_pairs), size=max_impostor_pairs, replace=False)
        all_pairs = [all_pairs[i] for i in idx]

    for c1, m1, c2, m2 in all_pairs:
        impostor.append(hamming_distance(c1, m1, c2, m2))

    return np.array(genuine), np.array(impostor)


# ======================================================
# 9. ROC
# ======================================================

def compute_roc(genuine, impostor, num_thresholds=400):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    fpr_list = []
    tpr_list = []

    for t in thresholds:
        tp = np.sum(genuine <= t)
        fn = np.sum(genuine > t)
        fp = np.sum(impostor <= t)
        tn = np.sum(impostor > t)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(thresholds), np.array(fpr_list), np.array(tpr_list)


def compute_auc(fpr, tpr):
    return float(np.trapezoid(tpr, fpr))


# ======================================================
# 10. K-MEANS
# ======================================================

def run_kmeans_on_distances(genuine, impostor, n_clusters=2, random_state=0):
    all_dist = np.concatenate([genuine, impostor])
    if len(all_dist) < n_clusters:
        return None

    X = all_dist.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.flatten()
    return centers


# ======================================================
# 11. GRAFICI
# ======================================================

def plot_balanced_histogram(genuine, impostor, out_path, random_state=0):
    set_plot_style()
    plt.figure()

    rng = np.random.default_rng(random_state)
    n = min(len(genuine), len(impostor))
    imp_sample = rng.choice(impostor, size=n, replace=False)

    all_scores = np.concatenate([genuine, imp_sample])
    xmin = max(0.0, np.min(all_scores) - 0.02)
    xmax = min(1.0, np.max(all_scores) + 0.02)
    bins = 20

    plt.hist(
        genuine,
        bins=bins,
        range=(xmin, xmax),
        density=True,
        alpha=0.7,
        label="Genuine (sottoinsieme)"
    )
    plt.hist(
        imp_sample,
        bins=bins,
        range=(xmin, xmax),
        density=True,
        alpha=0.7,
        label="Impostor (campionati)"
    )

    plt.xlabel("Distanza di Hamming")
    plt.ylabel("Densità")
    plt.title("Istogramma bilanciato delle distanze")
    plt.xlim(xmin, xmax)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_roc_curve(thresholds, fpr, tpr, out_path):
    set_plot_style()
    plt.figure()

    auc = compute_auc(fpr, tpr)

    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Curva ROC – Iris recognition")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.linspace(0, 1, 6))
    plt.yticks(np.linspace(0, 1, 6))
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_kmeans_centroids(centers, out_path):
    if centers is None:
        return
    set_plot_style()
    plt.figure()

    x = np.arange(len(centers))
    plt.bar(x, centers, width=0.6)

    plt.xticks(x, [f"Cluster {k}" for k in x])
    plt.ylabel("Centroide (distanza media)")
    plt.title("Centroidi dei cluster K-Means sulle distanze")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ======================================================
# 12. SALVATAGGIO CSV
# ======================================================

def save_csv_results(genuine, impostor, thresholds, fpr, tpr, centers):
    ensure_dir(CSV_DIR)

    # 1) Distanze genuine
    genuine_path = os.path.join(CSV_DIR, "genuine_distances.csv")
    np.savetxt(
        genuine_path,
        genuine,
        delimiter=",",
        header="genuine_distance",
        comments=""
    )

    # 2) Distanze impostor
    impostor_path = os.path.join(CSV_DIR, "impostor_distances.csv")
    np.savetxt(
        impostor_path,
        impostor,
        delimiter=",",
        header="impostor_distance",
        comments=""
    )

    # 3) Curva ROC: threshold, fpr, tpr
    roc_path = os.path.join(CSV_DIR, "roc_curve.csv")
    roc_data = np.column_stack([thresholds, fpr, tpr])
    np.savetxt(
        roc_path,
        roc_data,
        delimiter=",",
        header="threshold,fpr,tpr",
        comments=""
    )

    # 4) Centroidi K-Means (se presenti)
    if centers is not None:
        centers_path = os.path.join(CSV_DIR, "kmeans_centers.csv")
        with open(centers_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster", "center"])
            for idx, c in enumerate(centers):
                writer.writerow([idx, c])


# ======================================================
# 13. MAIN
# ======================================================

if __name__ == "__main__":
    ensure_dir(EXPORT_DIR)
    ensure_dir(CSV_DIR)  # <--- assicura la cartella CSV

    print("Carico e processo il dataset...")
    codes = compute_all_codes(DATASET_PATH)

    if len(codes) < 2:
        raise SystemExit("Servono almeno due persone con immagini valide.")

    print("Calcolo distanze genuine e impostor...")
    genuine, impostor = compute_distances(codes)
    print(f"Numero confronti genuine : {len(genuine)}")
    print(f"Numero confronti impostor: {len(impostor)}")

    # ROC
    thresholds, fpr, tpr = compute_roc(genuine, impostor, num_thresholds=400)

    # K-Means
    centers = run_kmeans_on_distances(
        genuine, impostor, n_clusters=2, random_state=0
    )

    # GRAFICI
    plot_balanced_histogram(
        genuine,
        impostor,
        os.path.join(EXPORT_DIR, "01_hist_balanced.png")
    )

    plot_roc_curve(
        thresholds,
        fpr,
        tpr,
        os.path.join(EXPORT_DIR, "03_roc_curve.png")
    )

    plot_kmeans_centroids(
        centers,
        os.path.join(EXPORT_DIR, "04_kmeans_centroids.png")
    )

    # CSV
    save_csv_results(genuine, impostor, thresholds, fpr, tpr, centers)

    print(f"Grafici creati in: {EXPORT_DIR}")
    print(f"File CSV creati in: {CSV_DIR}")