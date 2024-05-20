import random

import polars as pl
import seaborn as sns

sns.set_style("whitegrid")
# random.seed(42)
# pl.set_random_seed(42)


def make_df(
    n_speaker: int,
    n_sentence: int,
    n_model: int,
) -> pl.DataFrame:
    speaker_list = [i for i in range(n_speaker)]
    sentence_list = [i for i in range(n_sentence)]
    sentence_list = random.sample(sentence_list, len(sentence_list))
    model_list = [i for i in range(n_model)]

    chunk_size_list = [n_sentence // n_model for _ in range(n_model)]
    remainder = n_sentence % n_model
    add_index_list = random.sample([i for i in range(len(chunk_size_list))], remainder)
    for add_index in add_index_list:
        chunk_size_list[add_index] += 1

    sentence_list_split = []
    chunk_size_prev = 0
    for chunk_size in chunk_size_list:
        sentence_list_split.append(
            sentence_list[chunk_size_prev : chunk_size_prev + chunk_size]
        )
        chunk_size_prev += chunk_size

    sentence_group_tuple_list = []
    for sentence_group, sentece_list_ in enumerate(sentence_list_split):
        for sentence in sentece_list_:
            sentence_group_tuple_list.append((sentence, sentence_group))

    data = {
        "speaker": [],
        "sentence": [],
        "sentence_group": [],
        "model": [],
    }  # type: ignore
    for speaker in speaker_list:
        for sentence, sentence_group in sentence_group_tuple_list:
            for model in model_list:
                data["speaker"].append(speaker)
                data["sentence"].append(sentence)
                data["sentence_group"].append(sentence_group)
                data["model"].append(model)

    df = pl.DataFrame(data=data)
    return df


def main():
    n_speaker = 4
    n_sentence = 53
    n_model = 5
    n_ans_per_sample = 3
    n_trial = int(n_speaker * n_model * n_ans_per_sample)
    df = make_df(
        n_speaker=n_speaker,
        n_sentence=n_sentence,
        n_model=n_model,
    )
    df = df.with_columns(pl.lit(0).alias("n_selected"))

    for trial in range(n_trial):
        if df.select(pl.col("n_selected").n_unique()).to_numpy().reshape(-1)[0] == 1:
            df_cand = df
        else:
            n_selected_max = (
                df.select(pl.col("n_selected").max()).to_numpy().reshape(-1)
            )
            df_cand = df.filter(pl.col("n_selected") < n_selected_max)

        selected_data = {
            "sentence": [],
            "speaker": [],
            "model": [],
        }

        df_cand = df_cand.with_columns(
            ((pl.col("sentence_group") + trial) % n_model).alias("model_assign_id")
        )

        for row in (
            df.select(pl.col("sentence").unique())
            .sample(fraction=1, shuffle=True)
            .iter_rows()
        ):
            sentence = row[0]
            df_cand_sampled = df_cand.filter(
                (pl.col("sentence") == sentence)
                & (pl.col("model") == pl.col("model_assign_id"))
            ).sample(n=1, shuffle=True)
            selected_data["sentence"].append(sentence)
            selected_data["speaker"].append(
                df_cand_sampled.select(pl.col("speaker")).to_numpy().reshape(-1)[0]
            )
            selected_data["model"].append(
                df_cand_sampled.select(pl.col("model")).to_numpy().reshape(-1)[0]
            )

        df_selected_data = (
            pl.DataFrame(data=selected_data)
            .sample(fraction=1, shuffle=True)
            .with_columns(pl.lit(1).alias("is_selected"))
        )

        if (
            df_selected_data.select(pl.col("model").n_unique())
            .to_numpy()
            .reshape(-1)[0]
            != n_model
        ):
            raise ValueError("All models must be included at least once.")

        if df_selected_data.select(pl.col("sentence").n_unique()).to_numpy().reshape(
            -1
        )[0] != len(df_selected_data):
            raise ValueError("All sentences must be included once.")

        df = (
            df.join(
                other=df_selected_data, on=["sentence", "speaker", "model"], how="left"
            )
            .with_columns(pl.col("is_selected").fill_null(0))
            .with_columns(
                (pl.col("n_selected") + pl.col("is_selected")).alias("n_selected")
            )
            .drop("is_selected")
        )

    if len(df.group_by(pl.col("n_selected")).len()) != 1:
        raise ValueError("All combinations must be evaluated the same number of times.")


if __name__ == "__main__":
    main()