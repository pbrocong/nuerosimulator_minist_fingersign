import os
import pandas as pd

def read_split_by_write_vh(file_path: str):
    """엑셀에서 LTP/LTD 분리하여 반환"""
    if not os.path.exists(file_path):
        print(f"[에러] 파일을 찾을 수 없습니다: {file_path}")
        return None, None
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
        required = {"Conductance", "WriteVH", "PulseNum"}
        if not required.issubset(df.columns):
            print(f"[에러] 필수 열 {required} 중 일부가 없습니다.")
            return None, None
        ltp = df[df["WriteVH"] > 0].sort_values("PulseNum").reset_index(drop=True)
        ltd = df[df["WriteVH"] < 0].sort_values("PulseNum").reset_index(drop=True)
        if ltp.empty or ltd.empty:
            print("[에러] LTP 또는 LTD 데이터가 없습니다.")
            return None, None
        return ltp, ltd
    except Exception as e:
        print(f"[에러] 파일 읽기 실패: {e}")
        return None, None