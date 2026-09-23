# Model: `penta2himajin/tse-conv-tasnet-48k` (candidate C)

Repository: N/A
Checkpoint: N/A
Paper: N/A
License: N/A

## Availability

Can checkpoint actually be downloaded? **NO. This repository does not exist.**

Re-verified directly in this session (not assumed from an earlier finding):

```
$ curl -o /dev/null -w "HTTP:%{http_code}\n" https://huggingface.co/penta2himajin/tse-conv-tasnet-48k
HTTP:401

$ curl https://huggingface.co/api/models/penta2himajin/tse-conv-tasnet-48k
{"error":"Invalid username or password."}

$ curl https://huggingface.co/api/models?search=tse-conv-tasnet-48k
[]

$ curl https://huggingface.co/api/models?author=penta2himajin
[... 5 real repos by this account, none named "tse-conv-tasnet-48k" or
     related to TSE at all - a DeepFilterNet3 ONNX repo, an ECAPA-TDNN ONNX
     repo (both tagged with a non-standard "mellonella" library and zero
     downloads), and three unrelated LLM-translation-model repos ...]
```

The account `penta2himajin` is real and has public repositories, but none of them is a TSE model, and the specific claimed model ID returns a 401/not-found response and zero search matches by name. **The named candidate does not exist publicly.**

This also means all of the candidate's claimed properties (causal, streaming, native 48kHz, 480-sample chunks, ONNX, CPU, explicit state tensors, 192-D ECAPA speaker embedding) cannot be checked, because there is no file to check them against. These claimed properties are not confirmed to describe any real, existing model.

## Everything else in the template

Not applicable - there is nothing to inspect, download, hash, or run.

## What was actually done in this session

1. Re-ran the exact verification from an earlier phase of this project (direct HTTP fetch of the model page, direct Hugging Face API model lookup, a Hub search by name, and a listing of the account's actual public repositories) to confirm the earlier finding still holds and was not a transient error.
2. Confirmed the same account has two unrelated ONNX repos (`deepfilternet3-onnx`, `ecapa-tdnn-onnx`) tagged with a non-standard "mellonella" library and zero downloads - a low-provenance-signal account, consistent with the project's existing rule to not adopt uploads without a credible maintenance/provenance signal, independent of the specific 48kHz-TSE claim being false.

**Conclusion: this candidate cannot be evaluated further and is excluded from the comparison matrix as "does not exist," not as "rejected for cause."**
