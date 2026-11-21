import json
import math
import unicodedata
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
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def add_source_footer(fig: plt.Figure, source_text: str) -> None:
    fig.text(0.01, -0.02, source_text, fontsize=8.5, color="#555555", style="italic")

def save_figure_safely(fig: plt.Figure, out_path: Path, source_text: str = "", dpi: int = 300) -> None:
    if source_text:
        add_source_footer(fig, source_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    [+] Đã lưu biểu đồ: {out_path}")

def read_kpi_csv(proc_dir: Path, filename: str) -> pd.DataFrame:
    path = proc_dir / filename
    if not path.exists():
        print(f"    [!] Bỏ qua {filename} vì chưa tìm thấy tệp tại {path}.")
        return pd.DataFrame()
    return pd.read_csv(path)

def city_label_from_config(config: dict) -> str:
    return str(config.get("city", "")).replace("-", " ").title()

def plot_fig1_price_by_area(df_kpi1: pd.DataFrame, fig_dir: Path, city: str = "", last_snap: str = "") -> None:
    if df_kpi1.empty:
        return
    df_plot = df_kpi1.sort_values(by="gia_trung_vi_eur", ascending=True).reset_index(drop=True)
    area_col = "analysis_area" if "analysis_area" in df_plot.columns else df_plot.columns[0]

    fig_h = max(5.2, len(df_plot) * 0.34)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y_pos = list(range(len(df_plot)))

    bars = ax.barh(y_pos, df_plot["gia_trung_vi_eur"], height=0.65, color=PALETTE["primary"])

    if "so_phong_sach" in df_plot.columns:
        y_labels = [f"{a} (n={int(n)})" for a, n in zip(df_plot[area_col], df_plot["so_phong_sach"])]
    else:
        y_labels = df_plot[area_col].tolist()

    top_area = str(df_plot.iloc[-1][area_col])
    top_price = float(df_plot.iloc[-1]["gia_trung_vi_eur"])
    bot_area = str(df_plot.iloc[0][area_col])
    bot_price = float(df_plot.iloc[0]["gia_trung_vi_eur"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Giá thuê trung vị 1 đêm (EUR)", fontsize=10.5)
    ax.set_title(
        f"Giá trung vị cao nhất tại {top_area} ({top_price:.0f}€) và thấp nhất tại {bot_area} ({bot_price:.0f}€) ({city}, {last_snap})",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 2.5,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.0f}€",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE["primary"],
        )

    ax.set_xlim(0, top_price * 1.15)
    save_figure_safely(
        fig,
        fig_dir / "fig1_price_by_area.png",
        f"Nguồn: Inside Airbnb ({city}), mốc chụp {last_snap}. Chỉ tính chỗ ở sạch về giá.",
    )

def plot_fig1b_price_boxplot_log(
    proc_dir: Path,
    fig_dir: Path,
    city: str = "",
    last_snap: str = "",
    area_col: str = "analysis_area",
) -> None:
    listings_path = proc_dir / last_snap / "listings.parquet"
    if not listings_path.exists():
        return
    df = pd.read_parquet(listings_path)
    if "is_clean_price" in df.columns:
        df = df[df["is_clean_price"]].copy()
    if "price_num" not in df.columns or df.empty:
        return
    df = df[df["price_num"] > 0].copy()
    col = area_col if area_col in df.columns else "neighbourhood_cleansed"

    order = df.groupby(col)["price_num"].median().sort_values(ascending=True).index.tolist()
    counts = df[col].value_counts().to_dict()
    data_groups = [df.loc[df[col] == grp, "price_num"].values for grp in order]

    fig_h = max(5.2, len(order) * 0.34)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.boxplot(
        data_groups,
        vert=False,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["secondary"], "linewidth": 2.0},
        boxprops={"facecolor": "#d6e4f0", "edgecolor": PALETTE["primary"]},
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks([10, 20, 30, 50, 75, 100, 150, 200, 300, 400])
    ax.set_xlim(9, 500)
    ax.grid(axis="x", which="major", linestyle="--", alpha=0.4)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    med_all = float(df["price_num"].median())
    q25_all = float(df["price_num"].quantile(0.25))
    q75_all = float(df["price_num"].quantile(0.75))

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{g} (n={counts.get(g, 0)})" for g in order], fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Giá thuê 1 đêm (EUR — thang logarit, ẩn điểm ngoại lai > 1.5×IQR)", fontsize=10)
    ax.set_title(
        f"Một nửa số chỗ ở tại {city} có giá từ {q25_all:.0f}€ đến {q75_all:.0f}€/đêm (trung vị {med_all:.0f}€) ({last_snap})",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure_safely(
        fig,
        fig_dir / "fig1b_price_boxplot_log.png",
        f"Nguồn: Inside Airbnb ({city}), mốc chụp {last_snap} (n={len(df):,}). Vạch cam là giá trung vị.",
    )

def plot_fig2_occupancy_trend(df_kpi2: pd.DataFrame, fig_dir: Path, city: str = "", n_snapshots: int = 4) -> None:
    if df_kpi2.empty:
        return

    value_col = "ty_le_khong_kha_dung_pct"
    fig, ax = plt.subplots(figsize=(10, 5.2))
    is_city_all = df_kpi2["analysis_area"].astype(str).str.startswith("ALL")
    df_all = df_kpi2[is_city_all].sort_values("moc_thoi_gian")
    df_areas = df_kpi2[~is_city_all]

    for area_name, grp in df_areas.groupby("analysis_area"):
        grp_s = grp.sort_values("moc_thoi_gian")
        ax.plot(grp_s["moc_thoi_gian"], grp_s[value_col], marker="o", linewidth=1.1, alpha=0.35, linestyle="--", label=area_name)

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
        for idx_pt, (_, r) in enumerate(df_all.iterrows()):
            is_last_pt = idx_pt == len(df_all) - 1
            ax.annotate(
                f"{r[value_col]:.1f}%",
                (r["moc_thoi_gian"], r[value_col]),
                textcoords="offset points",
                xytext=(-6, 10) if is_last_pt else (0, 8),
                ha="center",
                fontweight="bold",
                color=PALETTE["negative"],
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.85},
            )
        occ_min = float(df_all[value_col].min())
        occ_max = float(df_all[value_col].max())
        title_str = f"Tỷ lệ ngày không trống (xấp xỉ qua lịch khóa) toàn {city} từ {occ_min:.1f}% đến {occ_max:.1f}% qua {n_snapshots} mốc"
    else:
        title_str = f"Diễn biến tỷ lệ ngày không trống theo khu vực qua {n_snapshots} mốc tại {city}"

    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Mốc chụp dữ liệu (Snapshot)", fontsize=10.5)
    ax.set_ylabel("Tỷ lệ ngày không trống (%)", fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8.5,
        ncol=1 if len(df_areas["analysis_area"].unique()) <= 10 else 2,
    )
    save_figure_safely(
        fig,
        fig_dir / "fig2_occupancy_trend.png",
        f"Nguồn: Inside Airbnb ({city}), {n_snapshots} mốc chụp. Lưu ý: gồm cả ngày chủ nhà tự khóa lịch.",
    )

def plot_fig3_review_activity(df_kpi3: pd.DataFrame, fig_dir: Path, city: str = "", n_snapshots: int = 4) -> None:
    if df_kpi3.empty:
        return
    df_plot = df_kpi3.sort_values("moc_thoi_gian").reset_index(drop=True)
    x_pos = list(range(len(df_plot)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)

    rev_start = int(df_plot["tong_review_tich_luy_tren_listings"].iloc[0])
    rev_end = int(df_plot["tong_review_tich_luy_tren_listings"].iloc[-1])
    zero_start = int(df_plot["so_phong_chua_co_review"].iloc[0])
    zero_end = int(df_plot["so_phong_chua_co_review"].iloc[-1])
    rev_word = "tăng" if rev_end >= rev_start else "giảm"
    zero_word = "tăng" if zero_end > zero_start else "giảm"

    bars = ax1.bar(x_pos, df_plot["tong_review_tich_luy_tren_listings"], width=0.42, color=PALETTE["primary"], alpha=0.88)
    ax1.set_ylabel("Tổng lượt đánh giá tích lũy (lượt)", color=PALETTE["primary"], fontsize=10, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_ylim(0, float(df_plot["tong_review_tich_luy_tren_listings"].max()) * 1.18)
    ax1.set_title(
        f"Tổng đánh giá {rev_word} từ {rev_start:,} đến {rev_end:,}; số phòng chưa có đánh giá {zero_word} từ {zero_start} đến {zero_end} ({city})",
        fontsize=10.8,
        fontweight="bold",
        pad=12,
    )

    for b in bars:
        h = b.get_height()
        ax1.text(
            b.get_x() + b.get_width() / 2,
            h * 1.02,
            f"{int(h):,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["primary"],
        )

    ax2.plot(x_pos, df_plot["so_phong_chua_co_review"], color=PALETTE["negative"], marker="o", markersize=8, linewidth=2.5)
    ax2.set_ylabel("Số phòng chưa có đánh giá (phòng)", color=PALETTE["negative"], fontsize=10, fontweight="bold")
    ax2.set_xlabel("Mốc chụp dữ liệu (Snapshot)", fontsize=10.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_plot["moc_thoi_gian"], fontsize=10)
    ax1.tick_params(labelbottom=True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_ylim(0, max(float(df_plot["so_phong_chua_co_review"].max()) * 1.38, 62))

    for idx, val in enumerate(df_plot["so_phong_chua_co_review"]):
        ax2.annotate(
            f"{int(val)} phòng",
            (idx, val),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontweight="bold",
            color=PALETTE["negative"],
        )

    fig.tight_layout()
    save_figure_safely(fig, fig_dir / "fig3_review_activity.png", f"Nguồn: Inside Airbnb ({city}), {n_snapshots} mốc chụp.")

def plot_fig3b_monthly_reviews(proc_dir: Path, fig_dir: Path, city: str = "", last_snap: str = "") -> None:
    reviews_path = proc_dir / last_snap / "reviews.parquet"
    if not reviews_path.exists():
        return

    df = pd.read_parquet(reviews_path)
    if "is_clean" in df.columns:
        df = df[df["is_clean"]].copy()

    date_col = "date_clean" if "date_clean" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    max_dt = df[date_col].max()
    month_end = max_dt + pd.offsets.MonthEnd(0)
    if max_dt.normalize() < month_end.normalize():
        df = df[df[date_col] < max_dt.replace(day=1)].copy()
        note_snap = f" Dữ liệu review đến {max_dt:%Y-%m-%d}; đã bỏ tháng {max_dt:%Y-%m} chưa trọn."
    else:
        note_snap = f" Dữ liệu review đến {max_dt:%Y-%m-%d}."
    if df.empty:
        return

    ts = df.set_index(date_col).sort_index()
    monthly_counts = ts.resample("MS").size()

    rolling_3m_full = monthly_counts.rolling(3, min_periods=3).mean()
    monthly_recent = monthly_counts.tail(36)
    rolling_recent = rolling_3m_full.tail(36)

    peak_month = monthly_recent.idxmax().strftime("%Y-%m")
    peak_val = int(monthly_recent.max())

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(
        monthly_recent.index,
        monthly_recent.values,
        color=PALETTE["primary"],
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label="Số lượt đánh giá trong mẫu / tháng",
    )
    ax.plot(
        rolling_recent.index,
        rolling_recent.values,
        color=PALETTE["secondary"],
        linestyle="--",
        linewidth=2.0,
        label="Trung bình trượt 3 tháng",
    )

    peak_idx = monthly_recent.idxmax()
    ax.annotate(
        f"Đỉnh: {peak_val} lượt\n({peak_month})",
        xy=(peak_idx, peak_val),
        xytext=(-110, -5),
        textcoords="offset points",
        fontweight="bold",
        fontsize=8.5,
        color=PALETTE["negative"],
        arrowprops={"arrowstyle": "->", "color": PALETTE["negative"], "lw": 1.2},
    )

    ax.set_title(
        f"Số lượt đánh giá theo tháng trong tập mẫu tại {city} đạt đỉnh vào {peak_month} ({peak_val} lượt)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Thời gian (Tháng)", fontsize=10.5)
    ax.set_ylabel("Số lượt đánh giá trong mẫu / tháng", fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)

    save_figure_safely(
        fig,
        fig_dir / "fig3b_monthly_reviews.png",
        f"Nguồn: Inside Airbnb ({city}), tập mẫu đánh giá cố định (n={len(df):,}).{note_snap}",
    )

def plot_fig4_host_professionalization(df_kpi4: pd.DataFrame, fig_dir: Path, city: str = "", last_snap: str = "") -> None:
    if df_kpi4.empty:
        return
    area_col = "analysis_area" if "analysis_area" in df_kpi4.columns else df_kpi4.columns[0]
    df_plot = df_kpi4.sort_values(by="ty_le_chuyen_nghiep_pct", ascending=True).reset_index(drop=True)

    fig_h = max(5.5, len(df_plot) * 0.34)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, fig_h), sharey=True)
    y_pos = list(range(len(df_plot)))

    if "tong_so_phong" in df_plot.columns:
        y_labels = [f"{a} (n={int(n)})" for a, n in zip(df_plot[area_col], df_plot["tong_so_phong"])]
    else:
        y_labels = df_plot[area_col].tolist()

    bars = ax1.barh(y_pos, df_plot["ty_le_chuyen_nghiep_pct"], color=PALETTE["primary"], height=0.62)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Tỷ lệ chỗ ở thuộc chủ nhà ≥ 2 chỗ ở (%)")
    ax1.set_title("Tỷ lệ chỗ ở do Chủ nhà nhiều phòng quản lý", fontweight="bold", fontsize=10.5)
    ax1.set_xlim(0, 112)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for b in bars:
        w = b.get_width()
        ax1.text(w + 1.2, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8, fontweight="bold")

    h = 0.36
    ax2.barh(
        [y + h / 2 for y in y_pos],
        df_plot["gia_median_chuyen_nghiep_eur"],
        height=h,
        color=PALETTE["secondary"],
        label="Chủ nhà ≥ 2 chỗ ở",
    )
    ax2.barh(
        [y - h / 2 for y in y_pos],
        df_plot["gia_median_ca_nhan_eur"],
        height=h,
        color=PALETTE["neutral"],
        label="Chủ nhà cá nhân (1 chỗ ở)",
    )
    ax2.set_xlabel("Giá thuê trung vị (EUR/đêm)")
    ax2.set_title("Giá trung vị: Chủ nhà nhiều phòng vs Cá nhân", fontweight="bold", fontsize=10.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    max_price_ax2 = float(df_plot[["gia_median_chuyen_nghiep_eur", "gia_median_ca_nhan_eur"]].max().max())
    ax2.set_xlim(0, max_price_ax2 * 1.25)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    pro_min = float(df_plot["ty_le_chuyen_nghiep_pct"].min())
    pro_max = float(df_plot["ty_le_chuyen_nghiep_pct"].max())

    fig.suptitle(
        f"Tỷ lệ chỗ ở thuộc chủ nhà có ≥ 2 phòng dao động từ {pro_min:.1f}% đến {pro_max:.1f}% tại {city} ({last_snap})",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout()
    save_figure_safely(
        fig,
        fig_dir / "fig4_host_professionalization.png",
        f"Nguồn: Inside Airbnb ({city}), mốc chụp {last_snap}. Đơn vị đếm: số chỗ ở (listing); nhóm thiếu dữ liệu để trống.",
    )

def plot_fig5_llm_sentiment_by_area(df_kpi5: pd.DataFrame, fig_dir: Path, city: str = "", last_snap: str = "") -> None:
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

    labels = [
        f"{a} (n={int(n)}{'*' if int(n) < 30 else ''})"
        for a, n in zip(df_plot["analysis_area"], df_plot["so_binh_luan_mau"])
    ]
    min_pos = float(pos_vals.min())
    max_pos = float(pos_vals.max())

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontweight="bold")
    ax.set_xlabel("Tỷ lệ phần trăm đánh giá (%)", fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.set_title(
        f"Tỷ lệ đánh giá tích cực đạt từ {min_pos:.1f}% đến {max_pos:.1f}% tại các khu vực ({city}, {last_snap})",
        fontsize=11.5,
        fontweight="bold",
        pad=28,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)

    for idx, (p, m, n) in enumerate(zip(pos_vals, mix_vals, neg_vals)):
        if p >= 8:
            ax.text(p / 2, idx, f"{p:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9)
        if m >= 5:
            ax.text(p + m / 2, idx, f"{m:.1f}%", va="center", ha="center", color="black", fontsize=8.5)

    has_small_n = (df_plot["so_binh_luan_mau"] < 30).any()
    note = " *Cảnh báo cỡ mẫu nhỏ (n < 30)." if has_small_n else ""
    save_figure_safely(
        fig,
        fig_dir / "fig5_llm_sentiment_by_area.png",
        f"Nguồn: Inside Airbnb ({city}), mẫu LLM cố định n={int(df_plot['so_binh_luan_mau'].sum())} (review đến 2025-09-13). "
        "Cảnh báo: Tỷ lệ Positive có thể cao hơn thực tế do LLM bỏ sót 50% (9/18) mẫu Mixed trên tập kiểm thử; tổng mẫu Negative chỉ có 11 lượt.",
    )

def plot_fig6_llm_aspects_diverging(df_kpi6: pd.DataFrame, fig_dir: Path, city: str = "", last_snap: str = "") -> None:
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

    top_pos_row = df_plot.sort_values(by="ty_le_khen_pct", ascending=False).iloc[0]
    top_neg_row = df_plot.sort_values(by="ty_le_che_pct", ascending=False).iloc[0]

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_plot["khia_canh"].str.upper(), fontweight="bold", fontsize=10)
    ax.set_xlabel(f"Tỷ lệ xuất hiện trong {n_reviews:,} đánh giá hợp lệ (% Chê <--- 0 ---> % Khen)", fontsize=10.5)
    ax.set_title(
        f"{top_pos_row['khia_canh'].capitalize()} được khen nhiều nhất (+{top_pos_row['ty_le_khen_pct']:.1f}%), "
        f"{top_neg_row['khia_canh'].lower()} bị chê nhiều nhất (-{top_neg_row['ty_le_che_pct']:.1f}%) ({city})",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    for b in bars_pos:
        w = b.get_width()
        if w > 0:
            ax.text(
                w + 0.8,
                b.get_y() + b.get_height() / 2,
                f"+{w:.1f}%",
                va="center",
                color=PALETTE["positive"],
                fontweight="bold",
                fontsize=9,
            )
    for b in bars_neg:
        w = b.get_width()
        if w < 0:
            ax.text(
                w - 0.8,
                b.get_y() + b.get_height() / 2,
                f"{w:.1f}%",
                va="center",
                ha="right",
                color=PALETTE["negative"],
                fontweight="bold",
                fontsize=9,
            )

    max_pos = max(float(pos_pct.max()), 10.0) * 1.18
    max_neg = min(float(neg_pct.min()) * 2.4, -15.0)
    ax.set_xlim(max_neg, max_pos)
    save_figure_safely(
        fig,
        fig_dir / "fig6_llm_aspects_diverging.png",
        f"Nguồn: Inside Airbnb ({city}), trích xuất LLM trên tập mẫu đánh giá cố định (n={n_reviews:,}, review đến 2025-09-13).",
    )

def load_geojson(geojson_path: Path) -> dict | None:
    if not geojson_path.exists():
        print(f"    [!] Bỏ qua bản đồ vì chưa có {geojson_path}.")
        return None
    with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Cùng bảng sửa lỗi mã hóa với src/qa.py (không import để vẫn chạy được `python src/viz.py`)
MACROMAN_FIX = {
    "\x9a": "ö", "\x8a": "ä", "\x9f": "ü", "§": "ß",
    "\x80": "Ä", "\x85": "Ö", "\x86": "Ü",
}

def normalize_area_name(name: object) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    text = str(name)
    for bad, good in MACROMAN_FIX.items():
        text = text.replace(bad, good)
    return unicodedata.normalize("NFC", text).strip().casefold()

def feature_rings(feature: dict) -> list[list]:
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates", [])
    if geom.get("type") == "Polygon":
        return [coords[0]] if coords else []
    if geom.get("type") == "MultiPolygon":
        return [poly[0] for poly in coords if poly]
    return []

def plot_fig7_price_map(
    df_kpi7: pd.DataFrame,
    geojson: dict | None,
    fig_dir: Path,
    city: str = "",
    last_snap: str = "",
) -> None:
    if df_kpi7.empty or geojson is None:
        return

    valid_df = df_kpi7[~df_kpi7["canh_bao_n_nho"]] if "canh_bao_n_nho" in df_kpi7.columns else df_kpi7
    if valid_df.empty:
        valid_df = df_kpi7
    # Chuẩn hóa tên ở cả hai phía: geojson gốc có thể còn lỗi mã hóa MacRoman như bảng listings
    price_by_name = {
        normalize_area_name(k): v
        for k, v in zip(valid_df["neighbourhood_cleansed"], valid_df["gia_trung_vi_eur"])
    }
    geo_names = {
        normalize_area_name((f.get("properties") or {}).get("neighbourhood"))
        for f in geojson.get("features", [])
    }
    unmatched = sorted(set(price_by_name) - geo_names)
    if unmatched:
        print(f"    [!] Quận có trong KPI 7 nhưng không khớp tên trong geojson: {unmatched}")

    vmin = float(valid_df["gia_trung_vi_eur"].min())
    vmax = float(valid_df["gia_trung_vi_eur"].max())
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    patches, facecolors, all_lats = [], [], []
    for feat in geojson.get("features", []):
        name = normalize_area_name((feat.get("properties") or {}).get("neighbourhood"))
        val = price_by_name.get(name, float("nan"))
        color = "#cfd8dc" if math.isnan(val) else cmap(norm(val))
        for ring in feature_rings(feat):
            patches.append(Polygon(ring, closed=True))
            facecolors.append(color)
            all_lats.extend(pt[1] for pt in ring)

    if not patches:
        print("    [!] Bỏ qua bản đồ vì geojson không có đa giác hợp lệ.")
        plt.close(fig)
        return

    collection = PatchCollection(patches, facecolor=facecolors, edgecolor="white", linewidth=0.6)
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect(1 / math.cos(math.radians(sum(all_lats) / len(all_lats))), adjustable="box")
    ax.set_xlabel("Kinh độ")
    ax.set_ylabel("Vĩ độ")

    top_q_row = valid_df.sort_values(by="gia_trung_vi_eur", ascending=False).iloc[0]
    ax.set_title(
        f"Giá thuê trung vị theo quận tại {city} — cao nhất ở {top_q_row['neighbourhood_cleansed']} ({top_q_row['gia_trung_vi_eur']:.0f}€) ({last_snap})",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    mappable = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.8)
    cbar.set_label("Giá trung vị (EUR/đêm)")
    save_figure_safely(
        fig,
        fig_dir / "fig7_price_map.png",
        f"Nguồn: Inside Airbnb ({city}), mốc {last_snap}. Vùng tô xám là quận thiếu dữ liệu hoặc n < 10. Lưu ý: diện tích quận không phản ánh số lượng phòng.",
    )

def run_viz(config_path: str | Path = "configs/vienna.yaml") -> None:
    config = load_config(config_path)
    city = city_label_from_config(config)
    snapshots = config.get("snapshots", [])
    last_snap = str(snapshots[-1]) if snapshots else ""
    area_col = config.get("kpi", {}).get("area_col", "analysis_area")
    proc_dir = Path("data/processed")
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_fig1_price_by_area(read_kpi_csv(proc_dir, "kpi_1_price_by_area.csv"), fig_dir, city, last_snap)
    plot_fig1b_price_boxplot_log(proc_dir, fig_dir, city, last_snap, area_col)
    plot_fig2_occupancy_trend(read_kpi_csv(proc_dir, "kpi_2_occupancy_trend.csv"), fig_dir, city, len(snapshots))
    plot_fig3_review_activity(read_kpi_csv(proc_dir, "kpi_3_review_activity_trend.csv"), fig_dir, city, len(snapshots))
    plot_fig3b_monthly_reviews(proc_dir, fig_dir, city, last_snap)
    plot_fig4_host_professionalization(read_kpi_csv(proc_dir, "kpi_4_host_professionalization.csv"), fig_dir, city, last_snap)
    plot_fig5_llm_sentiment_by_area(read_kpi_csv(proc_dir, "kpi_5_llm_sentiment_by_area.csv"), fig_dir, city, last_snap)
    plot_fig6_llm_aspects_diverging(read_kpi_csv(proc_dir, "kpi_6_llm_aspects_summary.csv"), fig_dir, city, last_snap)

    geojson = load_geojson(Path("data/raw") / last_snap / "neighbourhoods.geojson")
    plot_fig7_price_map(read_kpi_csv(proc_dir, "kpi_7_price_by_neighbourhood.csv"), geojson, fig_dir, city, last_snap)

    print("\n[+] Hoàn tất vẽ biểu đồ! Tất cả ảnh PNG đã được lưu tại thư mục figures/")

if __name__ == "__main__":
    print("-" * 80)
    print("[Bước 7] Vẽ bộ biểu đồ tĩnh lưu vào figures/...")
    print("-" * 80)
    run_viz()
