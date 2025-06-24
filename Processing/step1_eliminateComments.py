with open("../data/dataset_credit_cards", "r") as file:
    lines = file.readlines()

data_lines = [line for line in lines if not (line.strip().startswith('%')) 
                                        and not (line.strip().startswith('@')) 
                                        and not (line.strip() == '')]


column_names = ["CUST_ID", "BALANCE", "BALANCE_FREQUENCY",
                "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
                "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
                "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY",
                "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT",
                "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT",
                "TENURE"]

with open("../data/dataset_credit_cards_clean.csv", "w") as file:
    file.write(','.join(column_names) + '\n')
    file.writelines(data_lines)