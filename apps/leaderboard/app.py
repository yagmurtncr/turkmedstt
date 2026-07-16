from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_ID = "turkmedstt/turkish-asr-benchmark"

DISPLAY_COLUMNS = {
    "rank": "Sıra",
    "model": "Model",
    "backend": "Altyapı",
    "clips": "Klip",
    "mean_wer": "WER ↓",
    "mean_cer": "CER ↓",
    "mean_rtf": "RTF ↓",
    "mean_ascs_text": "ASCS ↑",
    "std_ascs_text": "ASCS σ ↓",
    "ascs_wer_correlation": "ASCS-WER r",
    "mean_ref_sentiment": "Ref. duygu",
    "mean_hyp_sentiment": "Hip. duygu",
    "mean_sentiment_drift": "Duygu kayması ↓",
}

SORT_COLUMNS = {
    "WER": ("mean_wer", True),
    "CER": ("mean_cer", True),
    "RTF": ("mean_rtf", True),
    "ASCS": ("mean_ascs_text", False),
    "ASCS standart sapması": ("std_ascs_text", True),
    "Duygu kayması": ("mean_sentiment_drift", True),
}

CSS = """
.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
}
.title-block {
    border-bottom: 1px solid #d7dbe0;
    padding: 8px 0 18px;
    margin-bottom: 12px;
}
.title-block h1 {
    font-size: 30px !important;
    line-height: 1.2 !important;
    letter-spacing: 0 !important;
    margin: 0 0 8px !important;
}
.summary-band {
    border-top: 3px solid #087f5b;
    border-bottom: 1px solid #d7dbe0;
    padding: 12px 0 8px;
    margin-bottom: 14px;
}
.summary-band table {
    width: 100%;
}
.summary-band td {
    text-align: center !important;
    font-size: 15px;
}
.metric-note {
    border-left: 3px solid #d97706;
    padding-left: 12px;
}
"""


def load_csv(filename: str) -> pd.DataFrame:
    path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=filename,
            repo_type="dataset",
        )
    )
    return pd.read_csv(path)


leaderboard = load_csv("summary/leaderboard.csv")
source_breakdown = load_csv("summary/source_breakdown.csv")

NUMERIC_COLUMNS = [
    "mean_wer",
    "mean_cer",
    "mean_rtf",
    "mean_ascs_text",
    "std_ascs_text",
    "ascs_wer_correlation",
    "mean_ref_sentiment",
    "mean_hyp_sentiment",
    "mean_sentiment_drift",
]


def present(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[list(DISPLAY_COLUMNS)].rename(columns=DISPLAY_COLUMNS).copy()
    numeric_display = [DISPLAY_COLUMNS[column] for column in NUMERIC_COLUMNS]
    result[numeric_display] = result[numeric_display].round(4)
    return result


def filter_leaderboard(
    query: str,
    backend: str,
    sort_metric: str,
) -> pd.DataFrame:
    frame = leaderboard.copy()
    query = (query or "").strip()
    if query:
        frame = frame[
            frame["model"].str.contains(query, case=False, regex=False, na=False)
        ]
    if backend != "Tümü":
        frame = frame[frame["backend"] == backend]
    column, ascending = SORT_COLUMNS[sort_metric]
    frame = frame.sort_values([column, "mean_wer"], ascending=[ascending, True])
    return present(frame)


def filter_source(source: str, metric: str) -> pd.DataFrame:
    frame = source_breakdown.copy()
    if source != "Tümü":
        frame = frame[frame["source"] == source]
    sort_column = {
        "WER": "mean_wer",
        "CER": "mean_cer",
        "RTF": "mean_rtf",
    }[metric]
    frame = frame.sort_values([sort_column, "model"])
    frame = frame.rename(
        columns={
            "model": "Model",
            "backend": "Altyapı",
            "source": "Kaynak",
            "clips": "Klip",
            "mean_wer": "WER ↓",
            "mean_cer": "CER ↓",
            "mean_rtf": "RTF ↓",
        }
    )
    for column in ["WER ↓", "CER ↓", "RTF ↓"]:
        frame[column] = frame[column].round(4)
    return frame


backend_choices = ["Tümü", *sorted(leaderboard["backend"].unique().tolist())]
source_choices = ["Tümü", *sorted(source_breakdown["source"].unique().tolist())]

with gr.Blocks(css=CSS, title="TurkMedSTT Türkçe ASR Liderlik Tablosu") as demo:
    gr.Markdown(
        """
        # TurkMedSTT Türkçe ASR Liderlik Tablosu

        20 modelin 1.060 genel Türkçe konuşma klibindeki WER, CER, hız ve
        AcoSemantic sonuçları.
        """,
        elem_classes=["title-block"],
    )

    gr.Markdown(
        """
        | Model | Klip | Süre | Veri kaynağı | Ana metrik |
        |---:|---:|---:|---:|---:|
        | **20** | **1.060** | **105 dk** | **3** | **WER + CER + RTF + ASCS** |
        """,
        elem_classes=["summary-band"],
    )

    with gr.Tab("Genel sıralama"):
        with gr.Row():
            query = gr.Textbox(
                label="Model ara",
                placeholder="whisper, wav2vec2, seamless...",
            )
            backend = gr.Dropdown(
                choices=backend_choices,
                value="Tümü",
                label="Altyapı",
            )
            sort_metric = gr.Dropdown(
                choices=list(SORT_COLUMNS),
                value="WER",
                label="Sırala",
            )
        main_table = gr.Dataframe(
            value=present(leaderboard),
            interactive=False,
            wrap=False,
            show_search="filter",
            max_height=650,
            label="Tüm sonuçlar",
        )
        for control in [query, backend, sort_metric]:
            control.change(
                filter_leaderboard,
                inputs=[query, backend, sort_metric],
                outputs=main_table,
            )

    with gr.Tab("Kaynak bazında"):
        with gr.Row():
            source = gr.Dropdown(
                choices=source_choices,
                value="Tümü",
                label="Kaynak",
            )
            source_metric = gr.Dropdown(
                choices=["WER", "CER", "RTF"],
                value="WER",
                label="Sırala",
            )
        source_table = gr.Dataframe(
            value=filter_source("Tümü", "WER"),
            interactive=False,
            wrap=False,
            show_search="filter",
            max_height=650,
            label="Kaynak bazlı sonuçlar",
        )
        source.change(
            filter_source,
            inputs=[source, source_metric],
            outputs=source_table,
        )
        source_metric.change(
            filter_source,
            inputs=[source, source_metric],
            outputs=source_table,
        )

    with gr.Tab("Metrik açıklamaları"):
        gr.Markdown(
            """
            | Metrik | Yön | Hesaplama ve yorum |
            |---|---:|---|
            | WER | Düşük | `(değiştirme + silme + ekleme) / referans kelime sayısı`. `0.1345`, yaklaşık %13,45 kelime hatasıdır. |
            | CER | Düşük | Aynı edit hesabının karakter düzeyinde uygulanmasıdır. Türkçe ek ve yazım farklılıklarını WER'den daha ayrıntılı gösterebilir. |
            | RTF | Düşük | `işleme süresi / ses süresi`. `0.10`, yaklaşık gerçek zamanın 10 katı hız demektir. Donanım ve backend'e bağlıdır. |
            | ASCS_text | Yüksek | `1 - abs(referans duygu skoru - hipotez duygu skoru)`. Metinsel duygu-anlam yönünün korunmasını ölçer. |
            | ASCS σ | Düşük | Klip düzeyindeki ASCS değerlerinin popülasyon standart sapmasıdır. Düşük değer daha tutarlı davranış gösterir. |
            | ASCS-WER r | Negatif beklenir | Klip düzeyindeki WER ve ASCS arasındaki Pearson korelasyonudur. Daha negatif değer, hata arttıkça ASCS'nin düşme eğilimini gösterir. |
            | Ref. duygu | Bağlamsal | Referans metinlerin ortalama duygu skorudur; model başarı metriği değil, test metinlerinin dağılım bilgisidir. |
            | Hip. duygu | Bağlamsal | Model transkripsiyonlarının ortalama duygu skorudur. Referans ortalamasıyla birlikte sistematik kaymayı gösterir. |
            | Duygu kayması | Düşük | Referans ve hipotez duygu skorları arasındaki ortalama mutlak farktır. Bu tanımda `1 - ortalama ASCS` ile aynıdır. |

            WER ve CER, model çok sayıda fazladan kelime veya karakter eklediğinde
            klip düzeyinde `1.0` değerini aşabilir. Yüksek ASCS doğru kelimeler
            üretildiği anlamına gelmez; düşük RTF de yüksek doğruluk anlamına gelmez.
            Sonuçlar birlikte değerlendirilmelidir.

            ### SER domain uyumsuzluğu tanısı

            | Tanı | Sonuç |
            |---|---:|
            | Değerlendirilen klip | 199 |
            | Ortalama maksimum duygu güveni | 0.255 |
            | En yüksek güven | 0.270 |
            | Güveni 0.30 üzerinde olan klip | 0 |
            | Eşleşen/eşleşmeyen AUC | 0.446 |

            Acted-speech üzerinde eğitilmiş ses-duygu modeli nötr okuma konuşmasında
            güvenilir ayrım üretemedi. Bu nedenle ses tabanlı sonuç model sıralamasına
            katılmadı ve metin tabanlı ASCS kullanıldı.
            """,
            elem_classes=["metric-note"],
        )

    with gr.Tab("Yöntem ve kapsam"):
        gr.Markdown(
            """
            ### Değerlendirme seti

            | Kaynak | Klip | Benchmark içindeki rolü |
            |---|---:|---|
            | Common Voice Türkçe | 447 | Topluluk tarafından kaydedilmiş kısa ve çok konuşmacılı genel ifadeler |
            | ISSAI Türkçe | 453 | Farklı cümle ve konuşmacı özellikleri taşıyan okuma konuşması |
            | OpenSLR 106 Türkçe | 160 | Ayrı dağıtılan bir korpustan kısa Türkçe ifadeler |
            | **Toplam** | **1.060** | Yaklaşık 105 dakika genel Türkçe konuşma |

            Tüm yayımlanan modeller aynı 1.060 klip üzerinde değerlendirilmiştir.
            Genel sıralama klip-bazlı makro ortalama WER'e göre yapılır. Her klip,
            uzunluğundan bağımsız olarak ortalamaya aynı ağırlıkla girer.

            ### Backend ne demek?

            `transformers`, `openai-whisper`, `faster-whisper`,
            `transformers_ctc` ve `transformers_pipeline` modelin çalıştırıldığı
            çıkarım uygulamasını belirtir. Aynı model farklı backend, decoding veya
            yazılım sürümüyle farklı hız ve küçük doğruluk farkları gösterebilir.

            ### Neler yayımlanmadı?

            Kaynak sesler, referans transkriptler, model hipotezleri ve yerel dosya
            yolları bu repoda bulunmaz. Yalnız anonim klip kimlikleri ve türetilmiş
            metrikler yayımlanır. Kaynak veri setlerinin özgün lisansları geçerlidir.

            ### Sıralama nasıl yorumlanmalı?

            Ana sıra WER'e göredir. CER Türkçedeki karakter ve ek hatalarını, RTF hız
            maliyetini, AcoSemantic sütunları ise metinsel duygu-anlam yönünün ne
            kadar korunduğunu tamamlayıcı olarak gösterir. Tek bir sütun bütün
            kullanım senaryoları için mutlak kazanan belirlemez.
            """
        )

    gr.Markdown(
        """
        Sonuç dosyaları:
        [turkmedstt/turkish-asr-benchmark](https://huggingface.co/datasets/turkmedstt/turkish-asr-benchmark)

        Muhammed Kumcu ve Yağmur Tuncer; benchmark tasarımı, veri hazırlama,
        model koşuları, metrik analizi, dokümantasyon ve yayın çalışmalarını
        birlikte yürütmüştür.
        """
    )


if __name__ == "__main__":
    demo.launch()
