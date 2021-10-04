import os

import baostock as bs
import pandas as pd


def main():
    bs.login()
    for filename in os.scandir("../raw/"):
        if filename.name[0] == ".":
            continue
        symbol = filename.name[:-4]
        print(f"Downloading {symbol}")
        symbol_bs = symbol[:2].lower() + "." + symbol[2:]
        rs = bs.query_history_k_data_plus(
            symbol_bs,
            "date,time,open,close",
            start_date="2000-01-01",
            end_date="2021-09-29",
            frequency="5",
            adjustflag="3",
        )
        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        result.to_pickle("../temp/" + symbol + ".pkl")


if __name__ == "__main__":
    main()
