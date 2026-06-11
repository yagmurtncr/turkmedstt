# Turkish ASR Readability Post-Processor

Türkçe ASR çıktılarındaki büyük/küçük harf, noktalama ve güvenilir kelime
hatalarını düzelten, anlamı korumaya odaklı metin son-işleme projesi.

```text
ham ASR çıktısı -> daha okunabilir Türkçe metin
```

Model serbest metin üretmez. Tek checkpoint içindeki ayrı görev başlıklarıyla
yalnızca güveni yüksek düzeltmeleri uygular.

## Yayınlanan Modeller

| Model | Kullanım alanı | WER | CER | İyileşen / kötüleşen |
| --- | --- | ---: | ---: | ---: |
| [V1 General](https://huggingface.co/turkmedstt/turkish-asr-readability-postprocessor-v1) | Genel Türkçe ASR | `0.06179 -> 0.04277` | `0.00940 -> 0.00631` | `222 / 5` |
| [V2 Medical](https://huggingface.co/turkmedstt/turkish-medical-asr-readability-postprocessor-v2) | Sağlık alanı uyarlaması | `0.31106 -> 0.18352` | `0.05491 -> 0.03547` | `127 / 0` |

Her iki değerlendirmede de zaten doğru olan girdilerden bozulan satır sayısı
`0` olmuştur.

**Önerilen genel model:** V1 General.

V2 Medical kontrollü sağlık metni bozulmalarıyla eğitilip değerlendirilmiştir.
Gerçek sağlık seslerinden üretilmiş ASR çıktıları üzerinde henüz
doğrulanmamıştır ve klinik kararlar için kullanılmamalıdır.

## Mimari

Model, `ytu-ce-cosmos/turkish-mini-bert-uncased` encoderı üzerine kurulu yaklaşık
11,6 milyon parametreli çok görevli token editörüdür:

- büyük/küçük harf düzeltme;
- noktalama düzenleme;
- sınırlandırılmış, yüksek güvenli kelime değiştirme.

Bu yaklaşım yanlış içerik ekleme ve anlam değiştirme riskini azaltır. Ses
bilgisinde kaybolan kelimeleri geri getiremez ve ağır bozulmuş transkriptleri
tamamen düzeltemez.

## Kullanım

Model ağırlıkları ve kullanıma hazır `inference.py` dosyaları Hugging Face model
sayfalarında bulunmaktadır:

```powershell
pip install torch transformers
python inference.py "bugün hava çok güzel dışarı çıkalım"
```

Beklenen çıktı biçimi:

```text
Bugün hava çok güzel dışarı çıkalım.
```

## Kaynak Kod

| Dosya | Açıklama |
| --- | --- |
| `scripts/multitask_token_editor_model.py` | Çok görevli model mimarisi |
| `scripts/build_multitask_token_edit_dataset.py` | Eğitim etiketi oluşturma |
| `scripts/train_multitask_token_editor.py` | Model eğitimi |
| `scripts/infer_multitask_token_editor.py` | Yerel çıkarım |
| `scripts/evaluate_multitask_token_editor.py` | Kör değerlendirme |
| `scripts/analyze_multitask_token_editor.py` | Threshold ve hata analizi |
| `scripts/build_medical_synthetic_pairs.py` | Kontrollü sağlık verisi üretimi |

## Sonuçlar

Kabul edilen modellerin sonuçları ve temel sınırlamaları
[reports/RESULTS.md](reports/RESULTS.md) dosyasında özetlenmiştir.

Büyük veri setleri, ses dosyaları, model ağırlıkları, checkpointler, geçici
çıktılar ve başarısız deneyler bu temiz GitHub yayınında tutulmamaktadır.

## Sınırlamalar

- Gerçek kelime tanıma hatalarında düzeltme kapsamı halen sınırlıdır.
- Kişi, yer ve alan terimleri ek doğrulama gerektirir.
- V2 Medical gerçek sağlık ASR verisiyle doğrulanmamıştır.
- Yüksek riskli kullanımlarda insan kontrolü gereklidir.

## Lisans

Eğitim sürecinde farklı lisanslara sahip veri kaynakları kullanılmıştır.
Yeniden dağıtım veya ticari kullanım öncesinde kaynak veri lisansları ve
Hugging Face model kartları incelenmelidir.
