import json
import math
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI", "Tahoma"]
plt.rcParams["axes.unicode_minus"] = False

PALETTE = {
    "primary": "#1f4e78",
    "secondary": "#ed7d31",
    "positive": "#2e7d32",
    "mixed": "#f9a825",
    "negative": "#c62828",
    "neutral": "#607d8b",
}

def load_config(config_path: str | Path = "configs/vienna.yaml") -> dict:
    """Đọc tệp cấu hình YAML an toàn với bảng mã utf-8."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_figure_safely(fig: plt.Figure, out_path: Path, dpi: int = 300) -> None:
    """Lưu đối tượng Figure ra tệp ảnh PNG độ phân giải cao và giải phóng bộ nhớ."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    [+] Đã lưu biểu đồ: {out_path}")


def read_kpi_csv(proc_dir: Path, filename: str) -> pd.DataFrame:
    """Đọc tệp CSV kết quả KPI; nếu chưa tồn tại trả về DataFrame rỗng."""
    path = proc_dir / filename
    if not path.exists():
        print(f"    [!] Bỏ qua {filename} vì chưa tìm thấy tệp tại {path}.")
        return pd.DataFrame()
    return pd.read_csv(path)


def city_label_from_config(config: dict) -> str:
    """Tên thành phố hiển thị trên tiêu đề (lấy từ config, không ghi cứng)."""
    return str(config.get("city", "")).replace("-", " ").title()

def plot_fig1_price_by_area(df_kpi1: pd.DataFrame, fig_dir: Path, city: str = "", last_snap: str = "") -> None:
    """Hình 1: So sánh Giá trung vị và Giá trung bình (EUR) theo vùng phân tích ở mốc cuối."""
    if df_kpi1.empty:
        return
    df_plot = df_kpi1.sort_values(by="gia_trung_vi_eur", ascending=True).reset_index(drop=True)
    area_col = "analysis_area" if "analysis_area" in df_plot.columns else df_plot.columns[0]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = range(len(df_plot))
    bar_h = 0.38

    bars_med = ax.barh(
        [y + bar_h / 2 for y in y_pos],
        df_plot["gia_trung_vi_eur"],
        height=bar_h,
        color=PALETTE["primary"],
        label="Giá trung vị (Median - EUR)",
    )
    bars_mean = ax.barh(
        [y - bar_h / 2 for y in y_pos],
        df_plot["gia_trung_binh_eur"],
        height=bar_h,
        color=PALETTE["secondary"],
        alpha=0.85,
        label="Giá trung bình (Mean - EUR)",
    )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_plot[area_col], fontsize=10, fontweight="bold")
    ax.set_xlabel("Mức giá thuê 1 đêm (EUR)", fontsize=11)
    ax.set_title(f"KPI 1: So sánh Giá Trung vị và Trung bình theo Khu vực tại {city} ({last_snap})", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

    for bar in bars_med:
        w = bar.get_width()
        ax.text(w + 2, bar.get_y() + bar.get_height() / 2, f"{w:.0f}€", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["primary"])
    for bar in bars_mean:
        w = bar.get_width()
        ax.text(w + 2, bar.get_y() + bar.get_height() / 2, f"{w:.0f}€", va="center", fontsize=8.5, color=PALETTE["secondary"])

    ax.set_xlim(0, float(df_plot["gia_trung_binh_eur"].max()) * 1.15)
    save_figure_safely(fig, fig_dir / "fig1_price_by_area.png")


def plot_fig2_occupancy_trend(df_kpi2: pd.DataFrame, fig_dir: Path, city: str = "", n_snapshots: int = 4) -> None:
    """Hình 2: Xu hướng tỷ lệ ngày không khả dụng qua các mốc thời gian (chỉ báo gần đúng của mức lấp đầy)."""
    if df_kpi2.empty:
        return

    value_col = "ty_le_khong_kha_dung_pct"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    df_all = df_kpi2[df_kpi2["analysis_area"] == "ALL_VIENNA"].sort_values("moc_thoi_gian")
    df_areas = df_kpi2[df_kpi2["analysis_area"] != "ALL_VIENNA"]

    # Vẽ đường mờ cho 8 vùng A01-A08
    for area_name, grp in df_areas.groupby("analysis_area"):
        grp_s = grp.sort_values("moc_thoi_gian")
        ax.plot(grp_s["moc_thoi_gian"], grp_s[value_col], marker="o", linewidth=1.2, alpha=0.45, linestyle="--", label=area_name)

    # Vẽ đường đậm cho toàn thành phố
    if not df_all.empty:
        ax.plot(
            df_all["moc_thoi_gian"],
            df_all[value_col],
            marker="s",
            markersize=8,
            linewidth=3.0,
            color=PALETTE["negative"],
            label=f"Toàn {city} (Trung bình)",
        )
        for _, r in df_all.iterrows():
            ax.annotate(
                f"{r[value_col]:.1f}%",
                (r["moc_thoi_gian"], r[value_col]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontweight="bold",
                color=PALETTE["negative"],
            )

    ax.set_title(f"KPI 2: Xu hướng Tỷ lệ ngày không khả dụng qua {n_snapshots} Mốc chụp tại {city}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mốc chụp dữ liệu (Snapshot)", fontsize=11)
    ax.set_ylabel("Tỷ lệ ngày không khả dụng (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    save_figure_safely(fig, fig_dir / "fig2_occupancy_trend.png")


def plot_fig3_review_activity(df_kpi3: pd.DataFrame, fig_dir: Path, n_snapshots: int = 4) -> None:
    """Hình 3: Tăng trưởng tổng lượt đánh giá tích lũy và số phòng chưa có đánh giá qua các mốc."""
    if df_kpi3.empty:
        return
    df_plot = df_kpi3.sort_values("moc_thoi_gian").reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(9.5, 5))
    x_pos = range(len(df_plot))

    bars = ax1.bar(x_pos, df_plot["tong_review_tich_luy_tren_listings"], width=0.45, color=PALETTE["primary"], alpha=0.85, label="Tổng số review tích lũy")
    ax1.set_xticks(list(x_pos))
    ax1.set_xticklabels(df_plot["moc_thoi_gian"], fontsize=10)
    ax1.set_ylabel("Tổng số lượt đánh giá tích lũy", color=PALETTE["primary"], fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    # Chừa khoảng trống phía trên cột để nhãn số không bị cắt hoặc đè lên tiêu đề
    ax1.set_ylim(0, float(df_plot["tong_review_tich_luy_tren_listings"].max()) * 1.15)

    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h * 1.01, f"{int(h):,}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=PALETTE["primary"])

    ax2 = ax1.twinx()
    line_zero, = ax2.plot(x_pos, df_plot["so_phong_chua_co_review"], color=PALETTE["negative"], marker="o", markersize=8, linewidth=2.5, label="Số phòng chưa có review (0 review)")
    ax2.set_ylabel("Số phòng chưa có đánh giá (phòng)", color=PALETTE["negative"], fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(df_plot["so_phong_chua_co_review"].max() * 1.3, 60))

    for idx, val in enumerate(df_plot["so_phong_chua_co_review"]):
        ax2.annotate(f"{int(val)} phòng", (idx, val), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color=PALETTE["negative"])

    # Gộp chú thích của cả hai trục vào một legend
    ax1.legend([bars, line_zero], [bars.get_label(), line_zero.get_label()], loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True)

    plt.title(f"KPI 3: Động lực Tăng trưởng Đánh giá & Giảm số phòng mới qua {n_snapshots} Mốc chụp", fontsize=12, fontweight="bold", pad=14)
    save_figure_safely(fig, fig_dir / "fig3_review_activity.png")


def plot_fig4_host_professionalization(df_kpi4: pd.DataFrame, fig_dir: Path, city: str = "") -> None:
    """Hình 4: Tỷ lệ Chủ nhà chuyên nghiệp (>=2 phòng) và so sánh mức giá trung vị theo vùng."""
    if df_kpi4.empty:
        return
    area_col = "analysis_area" if "analysis_area" in df_kpi4.columns else df_kpi4.columns[0]
    df_plot = df_kpi4.sort_values(by="ty_le_chuyen_nghiep_pct", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    x_pos = range(len(df_plot))

    # Biểu đồ trái: Tỷ lệ phòng thuộc chủ nhà chuyên nghiệp
    bars = ax1.bar(x_pos, df_plot["ty_le_chuyen_nghiep_pct"], color=PALETTE["primary"], width=0.55)
    ax1.set_xticks(list(x_pos))
    ax1.set_xticklabels(df_plot[area_col], fontweight="bold")
    ax1.set_ylabel("Tỷ lệ phòng thuộc Multi-listing Host (%)")
    ax1.set_title("Tỷ trọng Chủ nhà Chuyên nghiệp (>=2 phòng)", fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    # Biểu đồ phải: So sánh giá trung vị giữa Chủ nhà chuyên nghiệp vs Cá nhân
    w = 0.36
    ax2.bar([x - w / 2 for x in x_pos], df_plot["gia_median_chuyen_nghiep_eur"].fillna(0), width=w, color=PALETTE["secondary"], label="Chủ nhà Chuyên nghiệp (>=2)")
    ax2.bar([x + w / 2 for x in x_pos], df_plot["gia_median_ca_nhan_eur"].fillna(0), width=w, color=PALETTE["neutral"], label="Chủ nhà Cá nhân (1 phòng)")
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(df_plot[area_col], fontweight="bold")
    ax2.set_ylabel("Giá thuê trung vị (EUR)")
    ax2.set_title("So sánh Giá Trung vị: Chuyên nghiệp vs Cá nhân", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend()

    fig.suptitle(f"KPI 4: Mức độ Chuyên nghiệp hóa Chủ nhà và Chiến lược Giá tại {city}", fontsize=13, fontweight="bold", y=1.02)
    save_figure_safely(fig, fig_dir / "fig4_host_professionalization.png")


def plot_fig5_llm_sentiment_by_area(df_kpi5: pd.DataFrame, fig_dir: Path) -> None:
    """Hình 5: Cơ cấu cảm xúc khách hàng từ LLM (Positive / Mixed / Negative) theo khu vực."""
    if df_kpi5.empty:
        return
    df_plot = df_kpi5.sort_values(by="ty_le_positive_pct", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = range(len(df_plot))

    pos_vals = df_plot["ty_le_positive_pct"]
    mix_vals = df_plot["ty_le_mixed_pct"]
    neg_vals = df_plot["ty_le_negative_pct"]

    ax.barh(y_pos, pos_vals, color=PALETTE["positive"], label="Tích cực (Positive)")
    ax.barh(y_pos, mix_vals, left=pos_vals, color=PALETTE["mixed"], label="Hỗn hợp (Mixed)")
    ax.barh(y_pos, neg_vals, left=pos_vals + mix_vals, color=PALETTE["negative"], label="Tiêu cực (Negative)")

    # Hiện cỡ mẫu n cạnh tên vùng để người đọc biết độ tin cậy
    labels = [f"{a} (n={int(n)})" for a, n in zip(df_plot["analysis_area"], df_plot["so_binh_luan_mau"])]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontweight="bold")
    ax.set_xlabel("Tỷ lệ phần trăm đánh giá (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_title("KPI 5: Cơ cấu Cảm xúc Khách hàng (Trích xuất bằng LLM) theo Khu vực", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)

    for idx, (p, m, n) in enumerate(zip(pos_vals, mix_vals, neg_vals)):
        if p >= 8:
            ax.text(p / 2, idx, f"{p:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9)
        if m >= 5:
            ax.text(p + m / 2, idx, f"{m:.1f}%", va="center", ha="center", color="black", fontsize=8.5)

    save_figure_safely(fig, fig_dir / "fig5_llm_sentiment_by_area.png")


def plot_fig6_llm_aspects_diverging(df_kpi6: pd.DataFrame, fig_dir: Path) -> None:
    """Hình 6: Biểu đồ phân kỳ tỷ lệ Khen (+) và Chê (-) trên các khía cạnh chuẩn từ LLM."""
    if df_kpi6.empty:
        return
    df_plot = df_kpi6.sort_values(by="tong_luot_nhac", ascending=True).reset_index(drop=True)
    n_reviews = int(df_plot["so_binh_luan_hop_le"].iloc[0]) if "so_binh_luan_hop_le" in df_plot.columns else 0

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    y_pos = range(len(df_plot))

    pos_pct = df_plot["ty_le_khen_pct"]
    neg_pct = -df_plot["ty_le_che_pct"]

    bars_pos = ax.barh(y_pos, pos_pct, color=PALETTE["positive"], height=0.55, label="Tỷ lệ Khen (%)")
    bars_neg = ax.barh(y_pos, neg_pct, color=PALETTE["negative"], height=0.55, label="Tỷ lệ Chê (%)")

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_plot["khia_canh"].str.upper(), fontweight="bold", fontsize=10)
    ax.set_xlabel(f"Tỷ lệ xuất hiện trong {n_reviews:,} đánh giá hợp lệ (% Chê <--- 0 ---> % Khen)", fontsize=11)
    ax.set_title(f"KPI 6: Cán cân Khen / Chê theo {len(df_plot)} Khía cạnh Dịch vụ (Trích xuất bởi LLM)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    for b in bars_pos:
        w = b.get_width()
        if w > 0:
            ax.text(w + 0.8, b.get_y() + b.get_height() / 2, f"+{w:.1f}%", va="center", color=PALETTE["positive"], fontweight="bold", fontsize=9)
    for b in bars_neg:
        w = b.get_width()
        if w < 0:
            ax.text(w - 0.8, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", ha="right", color=PALETTE["negative"], fontweight="bold", fontsize=9)

    max_pos = max(float(pos_pct.max()), 10.0) * 1.18
    max_neg = min(float(neg_pct.min()), -5.0) * 1.35
    ax.set_xlim(max_neg, max_pos)
    save_figure_safely(fig, fig_dir / "fig6_llm_aspects_diverging.png")


def load_geojson(geojson_path: Path) -> dict | None:
    """Đọc tệp geojson ranh giới khu vực; thiếu tệp thì trả về None (bỏ qua bản đồ)."""
    if not geojson_path.exists():
        print(f"    [!] Bỏ qua bản đồ vì chưa có {geojson_path}.")
        return None
    with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)


def feature_rings(feature: dict) -> list[list]:
    """Trả về danh sách vòng ngoài (ring) của một feature Polygon/MultiPolygon."""
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates", [])
    if geom.get("type") == "Polygon":
        return [coords[0]] if coords else []
    if geom.get("type") == "MultiPolygon":
        return [poly[0] for poly in coords if poly]
    return []


def plot_fig7_price_map(df_kpi7: pd.DataFrame, geojson: dict | None, fig_dir: Path, city: str = "", last_snap: str = "") -> None:
    """Hình 7: Bản đồ giá trung vị (EUR) theo neighbourhood_cleansed; quận thiếu dữ liệu tô xám."""
    if df_kpi7.empty or geojson is None:
        return
    price_by_name = dict(zip(df_kpi7["neighbourhood_cleansed"], df_kpi7["gia_trung_vi_eur"]))
    n_by_name = dict(zip(df_kpi7["neighbourhood_cleansed"], df_kpi7["so_phong_sach"]))

    fig, ax = plt.subplots(figsize=(8.5, 7))
    patches, values, all_lats = [], [], []
    for feat in geojson.get("features", []):
        name = (feat.get("properties") or {}).get("neighbourhood")
        for ring in feature_rings(feat):
            patches.append(Polygon(ring, closed=True))
            values.append(price_by_name.get(name, float("nan")))
            all_lats.extend(pt[1] for pt in ring)

    if not patches:
        print("    [!] Bỏ qua bản đồ vì geojson không có đa giác hợp lệ.")
        plt.close(fig)
        return

    collection = PatchCollection(patches, edgecolor="white", linewidth=0.6, cmap="viridis")
    collection.set_array(pd.Series(values).to_numpy())
    collection.set_facecolor([
        collection.cmap(collection.norm(v)) if not math.isnan(v) else "#cfd8dc" for v in values
    ])
    collection.set_clim(float(df_kpi7["gia_trung_vi_eur"].min()), float(df_kpi7["gia_trung_vi_eur"].max()))
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect(1 / math.cos(math.radians(sum(all_lats) / len(all_lats))))
    ax.set_xlabel("Kinh độ")
    ax.set_ylabel("Vĩ độ")
    ax.set_title(f"KPI 7: Giá thuê trung vị theo quận tại {city} ({last_snap}; xám = chưa có dữ liệu)", fontsize=12, fontweight="bold", pad=12)

    mappable = matplotlib.cm.ScalarMappable(cmap="viridis", norm=matplotlib.colors.Normalize(
        vmin=float(df_kpi7["gia_trung_vi_eur"].min()), vmax=float(df_kpi7["gia_trung_vi_eur"].max())))
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.8)
    cbar.set_label("Giá trung vị (EUR/đêm)")
    save_figure_safely(fig, fig_dir / "fig7_price_map.png")

def run_viz(config_path: str | Path = "configs/vienna.yaml") -> None:
    """Đọc các tệp KPI CSV và xuất biểu đồ PNG sắc nét vào thư mục figures/."""
    print("--- BẮT ĐẦU VẼ BỘ BIỂU ĐỒ TĨNH (VIZ PIPELINE) ---")
    config = load_config(config_path)
    city = city_label_from_config(config)
    snapshots = config.get("snapshots", [])
    last_snap = str(snapshots[-1]) if snapshots else ""
    proc_dir = Path("data/processed")
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_fig1_price_by_area(read_kpi_csv(proc_dir, "kpi_1_price_by_area.csv"), fig_dir, city, last_snap)
    plot_fig2_occupancy_trend(read_kpi_csv(proc_dir, "kpi_2_occupancy_trend.csv"), fig_dir, city, len(snapshots))
    plot_fig3_review_activity(read_kpi_csv(proc_dir, "kpi_3_review_activity_trend.csv"), fig_dir, len(snapshots))
    plot_fig4_host_professionalization(read_kpi_csv(proc_dir, "kpi_4_host_professionalization.csv"), fig_dir, city)
    plot_fig5_llm_sentiment_by_area(read_kpi_csv(proc_dir, "kpi_5_llm_sentiment_by_area.csv"), fig_dir)
    plot_fig6_llm_aspects_diverging(read_kpi_csv(proc_dir, "kpi_6_llm_aspects_summary.csv"), fig_dir)

    geojson = load_geojson(Path("data/raw") / last_snap / "neighbourhoods.geojson")
    plot_fig7_price_map(read_kpi_csv(proc_dir, "kpi_7_price_by_neighbourhood.csv"), geojson, fig_dir, city, last_snap)

    print("\n[+] HOÀN TẤT VẼ BIỂU ĐỒ! Tất cả ảnh PNG đã được lưu tại thư mục figures/")

if __name__ == "__main__":
    run_viz()
