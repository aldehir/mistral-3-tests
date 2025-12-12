# Mistral 3 Tests

Adapted from: https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512

## Usage

Call `run.py` with the desired test:

```console
$ python run.py test-1.py
```

Outputs to console and test-1.log.

## Caveats

Test 3 tries to fill up the context of 256K. I do not have a machine that can
run at that context, so I reduced it by a factor of 10. You'll need at least
`28k` context to run, `32k` to be safe.
