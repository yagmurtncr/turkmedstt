# Nihai Sonuçlar

Bu depoda yalnızca kabul edilen iki modelin sonuçları sunulmaktadır.

## V1 General

Genel Türkçe ASR çıktıları için önerilen modeldir.

| Metrik | Ham ASR | V1 General | Göreli iyileşme |
| --- | ---: | ---: | ---: |
| WER | `0.06179` | `0.04277` | `%30,8` |
| CER | `0.00940` | `0.00631` | `%32,9` |

- İyileştirilen satır: `222`
- Kötüleştirilen satır: `5`
- Zaten doğru girdiyi bozma: `0`

Model: https://huggingface.co/turkmedstt/turkish-asr-readability-postprocessor-v1

## V2 Medical

Sağlık alanına uyarlanmış araştırma modelidir.

| Metrik | Bozuk sağlık metni | V2 Medical | Göreli iyileşme |
| --- | ---: | ---: | ---: |
| WER | `0.31106` | `0.18352` | `%41,0` |
| CER | `0.05491` | `0.03547` | `%35,4` |

- İyileştirilen satır: `127`
- Kötüleştirilen satır: `0`
- Zaten doğru girdiyi bozma: `0`

Model: https://huggingface.co/turkmedstt/turkish-medical-asr-readability-postprocessor-v2

V2 Medical kontrollü sağlık metni bozulmalarında değerlendirilmiştir. Gerçek
sağlık seslerinden üretilmiş ASR çıktıları üzerinde doğrulanmadığı için klinik
olarak doğrulanmış bir model değildir.

## Ürün Kararı

- Genel Türkçe için `V1 General` kullanılmalıdır.
- Sağlık alanında `V2 Medical` yalnızca araştırma ve kontrollü pilotlarda
  kullanılmalıdır.
- Her iki model de serbest üretim yerine güvenli, sınırlandırılmış düzeltme
  yapacak şekilde tasarlanmıştır.
