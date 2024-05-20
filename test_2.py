import random
from collections import deque
from pathlib import Path

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
    n_trial = 60
    df = make_df(
        n_speaker=n_speaker,
        n_sentence=n_sentence,
        n_model=n_model,
    )
    df = df.with_columns(pl.lit(0).alias("n_selected"))
    df_speaker_model = (
        df.select(pl.col("speaker"), pl.col("model"))
        .unique(subset=["speaker", "model"], keep="first")
        .with_row_index(name="speaker_model_id")
        .cast(dtypes={"speaker_model_id": pl.Int64})
    )
    df = df.join(other=df_speaker_model, on=["speaker", "model"], how="left")

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
            "speaker_model_id": [],
            "speaker": [],
            "model": [],
        }
        model_que = deque(
            [
                model
                for model in df.select(pl.col("model").unique())
                .sample(fraction=1, shuffle=True)
                .to_numpy()
                .reshape(-1)
            ]
        )

        for row in (
            df.select(pl.col("sentence").unique())
            .sample(fraction=1, shuffle=True)
            .iter_rows()
        ):
            sentence = row[0]
            df_cand_filter_sentence = df_cand.filter(pl.col("sentence") == sentence)
            if len(model_que) != 0:
                n_iter = 0
                while True:
                    model_cand = model_que.popleft()
                    df_cand_filter_sentence_model = df_cand_filter_sentence.filter(
                        pl.col("model") == model_cand
                    )
                    if len(df_cand_filter_sentence_model) != 0:
                        df_cand_sampled = df_cand_filter_sentence_model.sample(
                            n=1, shuffle=True
                        )
                        break
                    model_que.append(model_cand)
                    n_iter += 1
                    if n_iter == len(model_que):
                        break
            else:
                df_cand_sampled = df_cand.filter(pl.col("sentence") == sentence).sample(
                    n=1, shuffle=True
                )

            selected_data["sentence"].append(sentence)
            selected_data["speaker_model_id"].append(
                df_cand_sampled.select(pl.col("speaker_model_id"))
                .to_numpy()
                .reshape(-1)[0]
            )
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

        save_dir = Path("./test_2_selected_results")
        save_dir.mkdir(parents=True, exist_ok=True)
        df_selected_data.write_csv(str(save_dir / f"{trial}.csv"))

        df = (
            df.join(
                other=df_selected_data, on=["sentence", "speaker_model_id"], how="left"
            )
            .with_columns(pl.col("is_selected").fill_null(0))
            .with_columns(
                (pl.col("n_selected") + pl.col("is_selected")).alias("n_selected")
            )
            .drop("is_selected")
        )

    print(df.group_by(pl.col("n_selected")).len())


if __name__ == "__main__":
    main()
