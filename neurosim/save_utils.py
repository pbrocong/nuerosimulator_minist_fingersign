import pandas as pd

def save_accuracy_to_excel(train_hist, test_hist, filename="mnist_accuracy_log.xlsx"):
    df = pd.DataFrame({"Epoch": list(range(1, len(train_hist)+1)),
                       "Train Accuracy (%)": train_hist,
                       "Test Accuracy (%)":  test_hist})
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"[INFO] 저장: {filename}")

def save_sign_mnist_results_to_excel(total_hist, gachon_hist, filename="sign_mnist_accuracy_log.xlsx"):
    data = {"Epoch": list(range(1, len(total_hist)+1)), "Total Accuracy (%)": total_hist}
    for ch, series in gachon_hist.items():
        data[f'"{ch}" Accuracy (%)'] = series
    pd.DataFrame(data).to_excel(filename, index=False, engine="openpyxl")
    print(f"[INFO] 저장: {filename}")