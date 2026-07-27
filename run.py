# from src.ml_engine.adapters.ieee_adapter import IEEEAdapter


# def main():

#     adapter = IEEEAdapter(

#         "datasets/raw/train_transaction.csv",

#         "datasets/raw/train_identity.csv"

#     )

#     adapter.process(
#         "datasets/normalized/ieee_normalized.parquet"
#     )


# if __name__ == "__main__":
#     main()

from src.ml_engine.training.train_paysim import main as train_paysim
from src.ml_engine.training.train_ieee import main as train_ieee



def main():

    print(
        "\n===== TRAINING PAYSIM ====="
    )

    train_paysim()


    print(
        "\n===== TRAINING IEEE ====="
    )

    train_ieee()



if __name__ == "__main__":
    main()