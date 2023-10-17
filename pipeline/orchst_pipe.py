import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import random
from sklearn.model_selection import train_test_split

EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class DataProcessor:
    def __init__(self, parquet_file_path):
        self.df = pd.read_parquet(parquet_file_path)

    def process_data(self):
        self.calculate_weighted_ratings()
        self.split_data()

    def calculate_weighted_ratings(self):
        df = self.df
        # Definição da métrica North Star: vezes que o usuário ouviu / quantidade de vezes ouvidas
        df['count_plays'] = df.groupby('id_tracks')['plays'].transform('sum')
        df['count_plays'] = df['count_plays'].fillna(1)
        C = df['plays'].mean()
        m = df['count_plays'].quantile(0.95)

        def weighted_rating(x):
            v = x['plays']
            R = x['count_plays']
            return (v / (v + m) * R) + (m / (m + v) * C)

        df['wr'] = df.apply(weighted_rating, axis=1)
        self.df = df.sort_values('wr', ascending=False)

    def split_data(self):
        interactions_train_df, interactions_test_df = train_test_split(self.df, stratify=self.df['user_id'], test_size=0.20, random_state=42)
        return interactions_train_df, interactions_test_df



class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recomendar os itens mais populares que o usuário ainda não viu.
        recommendations_df = self.popularity_df.sort_values('wr', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" é necessário no modo verbose')

            recommendations_df = recommendations_df.merge(self.items_df, how='left', left_on='id_tracks',
                                                          right_on='id_tracks')[['wr', 'id_tracks']]

        return recommendations_df


class ModelEvaluator:
    def __init__(self, interactions_train_indexed_df, interactions_test_indexed_df, df):
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df
        self.df = df

    def get_items_interacted(self, person_id, interactions_df):
        interacted_items = interactions_df.loc[person_id]['id_tracks']
        return set(interacted_items if isinstance(interacted_items, pd.Series) else [interacted_items])

    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = self.get_items_interacted(person_id, self.interactions_train_indexed_df)
        all_items = set(self.df['id_tracks'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def evaluate_model_for_user(self, model, person_id):
        interacted_values_testset = self.interactions_test_indexed_df.loc[person_id]
        if isinstance(interacted_values_testset['id_tracks'], pd.Series):
            person_interacted_items_testset = set(interacted_values_testset['id_tracks'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['id_tracks'])])
        interacted_items_count_testset = len(person_interacted_items_testset)

        person_recs_df = model.recommend_items(person_id, items_to_ignore=self.get_items_interacted(person_id, self.interactions_train_indexed_df), topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0

        for item_id in person_interacted_items_testset:
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, sample_size=100, seed=item_id % (2 ** 32))
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))
            valid_recs_df = person_recs_df[person_recs_df['id_tracks'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['id_tracks'].values
            hit_at_5, _ = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, _ = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count': hits_at_5_count,
                          'hits@10_count': hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        people_metrics = []
        for person_id in self.interactions_test_indexed_df.index.unique().values:
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())

        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df
    
    def print_metrics(self, model_name, metrics):
        print(f"Metrics for {model_name}:")
        print(f"Recall@5: {metrics['recall@5']:.2f}")
        print(f"Recall@10: {metrics['recall@10']:.2f}")
        print("\n")





def main():
    data_processor = DataProcessor('data/processed_recommendation_data.parquet')
    data_processor.process_data()
    interactions_train_indexed_df, interactions_test_indexed_df = data_processor.split_data()

    item_popularity_df = data_processor.df.groupby('id_tracks')['wr'].sum().sort_values(ascending=False).reset_index()


    popularity_model = PopularityRecommender(item_popularity_df, data_processor.df)
    
    print('Evaluating Popularity recommendation model...')
    model_evaluator = ModelEvaluator(interactions_train_indexed_df,
                                     interactions_test_indexed_df, data_processor.df)

if __name__ == '__main__':
    main()
