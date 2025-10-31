import pandas as pd

def load_instacart_data(data_path='./data/instacart/'):
    orders = pd.read_csv(f'{data_path}orders.csv')
    order_products_prior = pd.read_csv(f'{data_path}order_products__prior.csv')
    order_products_train = pd.read_csv(f'{data_path}order_products__train.csv')
    products = pd.read_csv(f'{data_path}products.csv')
    aisles = pd.read_csv(f'{data_path}aisles.csv')
    departments = pd.read_csv(f'{data_path}departments.csv')

    products = products.merge(aisles, on='aisle_id', how='left')
    products = products.merge(departments, on='department_id', how='left')

    order_products = order_products_prior.merge(orders[['order_id', 'user_id']], on='order_id')
    order_products = order_products.merge(
        products[['product_id', 'product_name', 'aisle', 'department']],
        on='product_id'
    )

    return {
        'orders': orders,
        'order_products': order_products,
        'order_products_train': order_products_train,
        'products': products,
        'aisles': aisles,
        'departments': departments
    }

def create_user_item_matrix(order_products, min_orders=5):
    user_counts = order_products['user_id'].value_counts()
    active_users = user_counts[user_counts >= min_orders].index
    df_filtered = order_products[order_products['user_id'].isin(active_users)]
    user_item = df_filtered.groupby(['user_id', 'product_id']).size().reset_index(name='purchase_count')
    
    return user_item, df_filtered