import struct
import pandas as pd

REC = struct.Struct('<176sf')  # total = 180 bytes

def iter_predictions(path):
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(REC.size)
            if not chunk:
                break
            if len(chunk) < REC.size:
                break  # partial tail
            raw_path, decision = REC.unpack(chunk)
            filepath = raw_path.split(b'\x00', 1)[0].decode('utf-8', errors='ignore')
            yield filepath, float(decision)

def get_predictions_df(path):
    preds = []
    for fp, dec in iter_predictions(path):
        preds.append({"rec":fp.split("/")[-1], "score":dec})
    return pd.DataFrame(preds)
    
def get_predictions(path):
    return list(iter_predictions(path))
