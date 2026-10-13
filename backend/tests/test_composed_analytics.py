import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.data_loader import DatasetLoader
from app.query_interpreter import QueryInterpreter
from app.main import app
import app.main as main

@pytest.fixture
def api(monkeypatch):
    loader = object.__new__(DatasetLoader)
    loader.sales = pd.DataFrame({
        'order_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'order_date': pd.to_datetime(['2023-03-01']*4 + ['2024-03-01']*4),
        'region': ['East', 'East', 'West', 'West']*2,
        'country': ['India']*8,
        'customer_id': ['A', 'B', 'C', 'D']*2,
        'product_name': ['Desk', 'Chair', 'Desk', 'Chair']*2,
        'revenue': [100., 50., 0., 0., 200., 75., 40., 20.],
        'profit': [10., 5., 0., 0., 20., 7., 4., 2.],
    })
    loader.targets = pd.DataFrame({'region': ['East', 'West'], 'month': ['2024-03', '2024-03'], 'target_revenue': [250., 100.]})
    loader.dictionary = {'metrics': {'revenue':'revenue','profit':'profit'}, 'dimensions':['region','country']}
    loader.queries = []
    monkeypatch.setattr(main, 'get_loader', lambda:loader)
    monkeypatch.setattr(QueryInterpreter, '_build_client', lambda self:None)
    monkeypatch.setattr(QueryInterpreter, '_backup_configuration', lambda self, provider:(None,''))
    monkeypatch.setattr(main.FeedbackStore, 'get_relevant', lambda self,q:[])
    def ask(query):
        response = TestClient(app).post('/query',json={'query':query})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload['is_supported'], payload
        return payload['result']
    return ask, loader

@pytest.mark.parametrize('phrase', ['AOV', 'Average order value', 'Mean order value'])
def test_ratio_changes_with_data(api, phrase):
    ask, loader = api
    result = ask(phrase+' by region in March 2024')
    assert result == [{'region':'East','avg_order_value':137.5}, {'region':'West','avg_order_value':30.}]
    loader.sales.loc[4,'revenue'] = 400.
    assert ask(phrase+' by region in March 2024')[0]['avg_order_value'] == 237.5

@pytest.mark.parametrize('month', ['March', 'Mar'])
def test_month_and_country_composition(api, month):
    ask, _ = api
    assert ask('Total sales in India for '+month+' 2024') == [{'revenue':335.}]
    assert ask('Total sales in India for '+month) == [{'revenue':485.}]

@pytest.mark.parametrize('metric,total', [('revenue',200.), ('profit',20.)])
def test_rank_then_reaggregate_uses_requested_metric(api, metric, total):
    ask, _ = api
    rows = ask('Total '+metric+' of top 1 customers per region in March 2024')
    assert rows[0] == {'region':'East',metric:total}
    assert len(rows) == 2

def test_singular_top_returns_one_per_partition(api):
    ask, _ = api
    rows = ask('Top product in each region in March 2024')
    assert len(rows) == 2
    assert all(row['product_name'] == 'Desk' for row in rows)

def test_yoy_keeps_prior_year_when_output_year_filtered(api):
    ask, _ = api
    rows = ask('YoY growth in revenue by region in 2024')
    east = next(row for row in rows if row['region']=='East')
    west = next(row for row in rows if row['region']=='West')
    assert east['previous_year_value'] == 150.
    assert east['yoy_growth'] == pytest.approx(125./150.)
    assert west['yoy_growth'] is None
    assert all(row['year']==2024 for row in rows)

def test_missing_prior_year_is_null(api):
    ask, loader = api
    loader.sales = loader.sales[loader.sales.order_date.dt.year == 2024]
    assert ask('Year over year growth in profit')[0]['yoy_growth'] is None

def test_target_filter_after_month_alignment(api):
    ask, _ = api
    rows = ask('Which region missed its target in March 2024?')
    assert len(rows) == 1
    assert rows[0]['region'] == 'West'
    assert rows[0]['actual'] == 60.
    assert rows[0]['target'] == 100.

@pytest.mark.parametrize('question,base', [
    ('Average order value by region', {'operation':'aggregation','metric':'revenue','group_by':['region']}),
    ('YoY growth in profit by region', {'operation':'aggregation','metric':'profit','group_by':['region']}),
    ('Revenue of top 2 customers per region', {'operation':'ranking','metric':'revenue','group_by':['region','customer_id'],'rank_dimension':'customer_id','partition_by':['region'],'limit_scope':'per_group','limit':2}),
])
def test_model_supplied_plans_use_composed_execution(api, monkeypatch, question, base):
    ask, _ = api
    monkeypatch.setattr(QueryInterpreter, '_build_client', lambda self:object())
    monkeypatch.setattr(QueryInterpreter, '_call_llm', lambda self, context:base)
    rows = ask(question)
    assert rows
    if 'order value' in question:
        assert rows[0]['avg_order_value'] == 106.25
    elif 'YoY' in question:
        assert next(row for row in rows if row['region']=='East' and row['year']==2024)['yoy_growth'] == pytest.approx(12./15.)
    else:
        assert rows[0] == {'region':'East','revenue':425.}

@pytest.mark.parametrize('question', ['Revenue of top 3 customers per region', 'Sales of the top three customers within each region'])
def test_reaggregation_paraphrases(api, question):
    ask, _ = api
    assert ask(question) == [{'region':'East','revenue':425.}, {'region':'West','revenue':60.}]

@pytest.mark.parametrize('question', [
    'Median revenue of top 2 customers per region',
    'Profit of top 2 customers by revenue per region',
    'Total revenue in March before 2024',
    'YoY growth in revenue before 2024',
])
def test_unrepresentable_combinations_are_rejected(api, question):
    response = TestClient(app).post('/query', json={'query':question})
    assert response.status_code in {200,422}
    if response.status_code == 200:
        assert response.json()['is_supported'] is False
