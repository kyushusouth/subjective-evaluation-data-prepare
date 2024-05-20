import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

sns.set_style("whitegrid")


def make_df(
    n_speaker: int,
    n_sentence: int,
    n_model: int,
) -> pl.DataFrame:
    speaker_list = [i for i in range(n_speaker)]
    sentence_list = [i for i in range(n_sentence)]
    sentence_list = random.sample(sentence_list, len(sentence_list))
    model_list = [i for i in range(n_model)]

    chunk_size_list = [n_sentence // n_speaker for _ in range(n_speaker)]
    remainder = n_sentence % n_speaker
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
    n_trial = 100
    use_weighted_prob = True
    speaker_list = [i for i in range(n_speaker)]
    df = make_df(
        n_speaker=n_speaker,
        n_sentence=n_sentence,
        n_model=n_model,
    )
    df = df.with_columns(pl.lit(0).alias("n_selected")).with_columns(
        (1 / (pl.col("n_selected") + 1)).alias("weight")
    )
    df_weight_sum = df.group_by(pl.col("sentence_group", "speaker", "sentence")).agg(
        pl.col("weight").sum().alias("weight_sum")
    )
    df = (
        df.join(
            other=df_weight_sum,
            on=["sentence_group", "speaker", "sentence"],
            how="left",
        )
        .with_columns((pl.col("weight") / pl.col("weight_sum")).alias("prob"))
        .drop("weight_sum")
    )

    for trial in range(n_trial):
        selected_data = {
            "speaker": [],
            "sentence_group": [],
            "sentence": [],
            "model": [],
        }
        speaker_list_shuffle = random.sample(speaker_list, len(speaker_list))

        for sentence_group, speaker in enumerate(speaker_list_shuffle):
            if use_weighted_prob:
                data = df.filter(
                    (pl.col("sentence_group") == sentence_group)
                    & (pl.col("speaker") == speaker)
                ).select(pl.col("sentence"), pl.col("prob"))
                sentence_model_list = []
                for row in data.select(pl.col("sentence").unique()).iter_rows():
                    sentence = row[0]
                    prob = (
                        data.filter(pl.col("sentence") == sentence)
                        .select(pl.col("prob"))
                        .to_numpy()
                        .reshape(-1)
                    )
                    model = np.random.choice(list(range(n_model)), p=prob)
                    sentence_model_list.append((sentence, model))
                for sentence, model in sentence_model_list:
                    selected_data["speaker"].append(speaker)
                    selected_data["sentence_group"].append(sentence_group)
                    selected_data["sentence"].append(sentence)
                    selected_data["model"].append(model)
            else:
                data = (
                    df.filter(
                        (pl.col("sentence_group") == sentence_group)
                        & (pl.col("speaker") == speaker)
                    )
                    .select(pl.col("sentence"))
                    .unique()
                )
                data = data.with_columns(
                    pl.lit(np.random.randint(0, n_model, (len(data),))).alias("model")
                )
                for row in data.iter_rows():
                    sentence = row[0]
                    model = row[1]
                    selected_data["speaker"].append(speaker)
                    selected_data["sentence_group"].append(sentence_group)
                    selected_data["sentence"].append(sentence)
                    selected_data["model"].append(model)

        df_selected_data = pl.DataFrame(data=selected_data)
        df_selected_data = df_selected_data.sample(
            n=len(df_selected_data), shuffle=True
        ).with_columns(pl.lit(1).alias("add"))
        df = (
            df.join(
                other=df_selected_data,
                on=["speaker", "sentence_group", "sentence", "model"],
                how="left",
            )
            .with_columns(
                (pl.col("n_selected") + pl.col("add").fill_null(0)).alias("n_selected"),
            )
            .drop("add")
            .with_columns((1 / (pl.col("n_selected") + 1)).alias("weight"))
        )
        df_weight_sum = df.group_by(
            pl.col("sentence_group", "speaker", "sentence")
        ).agg(pl.col("weight").sum().alias("weight_sum"))
        df = (
            df.join(
                other=df_weight_sum,
                on=["sentence_group", "speaker", "sentence"],
                how="left",
            )
            .with_columns((pl.col("weight") / pl.col("weight_sum")).alias("prob"))
            .drop("weight_sum")
        )

    df_n_selected_sum_by_speaker = df.group_by(pl.col("speaker")).agg(
        pl.col("n_selected").sum()
    )
    df_n_selected_sum_by_sentence = df.group_by(pl.col("sentence")).agg(
        pl.col("n_selected").sum()
    )
    df_n_selected_sum_by_model = df.group_by(pl.col("model")).agg(
        pl.col("n_selected").sum()
    )

    save_dir = Path("./fig")
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    sns.countplot(data=df, x="n_selected", ax=ax)
    fig.tight_layout()
    if use_weighted_prob:
        fig.savefig(str(save_dir / f"n_selected_weighted_{n_trial}.png"))
    else:
        fig.savefig(str(save_dir / f"n_selected_{n_trial}.png"))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    sns.barplot(data=df_n_selected_sum_by_speaker, x="speaker", y="n_selected", ax=ax)
    fig.tight_layout()
    if use_weighted_prob:
        fig.savefig(str(save_dir / f"n_selected_sum_by_speaker_weighted_{n_trial}.png"))
    else:
        fig.savefig(str(save_dir / f"n_selected_sum_by_speaker_{n_trial}.png"))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    sns.barplot(data=df_n_selected_sum_by_sentence, x="sentence", y="n_selected", ax=ax)
    fig.tight_layout()
    if use_weighted_prob:
        fig.savefig(
            str(save_dir / f"n_selected_sum_by_sentence_weighted_{n_trial}.png")
        )
    else:
        fig.savefig(str(save_dir / f"n_selected_sum_by_sentence_{n_trial}.png"))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    sns.barplot(data=df_n_selected_sum_by_model, x="model", y="n_selected", ax=ax)
    fig.tight_layout()
    if use_weighted_prob:
        fig.savefig(str(save_dir / f"n_selected_sum_by_model_weighted_{n_trial}.png"))
    else:
        fig.savefig(str(save_dir / f"n_selected_sum_by_model_{n_trial}.png"))


if __name__ == "__main__":
    main()
