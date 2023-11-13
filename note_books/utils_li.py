from langdetect import detect
import re
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import googlemaps
import folium
import numpy as np
import emoji
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from datetime import datetime
import plotly.express as px

model = SentenceTransformer('all-mpnet-base-v2')


def create_folium_map(latitude, longitude, map_title=None, height=500, width=800):
    # Calculate the average latitude and longitude for map center
    map_center = [sum(latitude) / len(latitude), sum(longitude) / len(longitude)]

    # Create a Folium map centered on the average latitude and longitude
    m = folium.Map(location=map_center, zoom_start=2, height=height, width=width)

    # Add markers for each latitude and longitude coordinate
    for lat, lon in zip(latitude, longitude):
        folium.Marker([lat, lon]).add_to(m)

    # Add title if provided
    if map_title:
        html = f'<h3 style="text-align: center;">{map_title}</h3>'
        title_popup = folium.Popup(html)
        m.get_root().html.add_child(title_popup)

    return m


def years_and_months_to_months(df, years_col, months_col, new_col_name):
    df[new_col_name] = df[years_col] * 12 + df[months_col]
    return df


def calculate_total_months_sum(df, id_col, months_col):
    total_months_sum = df.groupby(id_col)[months_col].sum().reset_index()
    return total_months_sum


def plot_columns(df, col1, col2):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col1], color='blue', label=col1, kde=True)
    sns.histplot(df[col2], color='red', label=col2, kde=True)
    plt.legend()
    plt.show()


def encode_string(text):
    embeddings = model.encode([text])
    return embeddings[0]


def get_embeddings(df, column):
    embeddings = model.encode(df[column].tolist())
    return embeddings


# Function to encode embeddings for a column
def encode_embeddings_2(column):
    return [model.encode(text) for text in column]


def plot_category_counts(df, column):
    category_counts = df[column].value_counts()

    plt.figure(figsize=(20, 10))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
    plt.title(column + ' Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=90)  # Rotate labels on x-axis for better visibility if they're long.
    plt.show()


def plot_numeric_distribution(df, column):
    plt.figure(figsize=(10, 8))
    sns.histplot(df[column], kde=True, bins=30, color='blue')
    plt.title('Distribution of ' + str(column))
    plt.xlabel('Numeric Column')
    plt.show()


def calculate_years_sum(df, id_col, years_col):
    years_sum = df.groupby(id_col)[years_col].sum().reset_index()
    return years_sum


def calculate_years(df, date_from, date_to, new_col_name):
    df[new_col_name] = df[date_to] - df[date_from]
    df.loc[(df[date_from] == 0) | (df[date_to] == 0), new_col_name] = 0
    return df


# get address given location string:
def geocode_addresses(df, address_column, api_key):
    gmaps = googlemaps.Client(key=api_key)

    def geocode_address(address):
        geocode_result = gmaps.geocode(address)
        if geocode_result and len(geocode_result) > 0:
            location = geocode_result[0]['geometry']['location']
            return pd.Series([location['lat'], location['lng']])
        else:
            return pd.Series([None, None])

    df[['latitude', 'longitude']] = df[address_column].apply(geocode_address)

    df["latitude"] = df["latitude"].fillna(0)
    df["longitude"] = df["longitude"].fillna(0)

    return df


def plot_country_heatmap(df, country_column, category_column, value_column):
    # Pivot the DataFrame to create a matrix of countries and values
    heatmap_data = df.pivot(index=country_column, columns=category_column, values=value_column)

    # Create the heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.1f', linewidths=0.5)
    plt.title('Heatmap of Countries')
    plt.xlabel('Category')
    plt.ylabel('Country')
    plt.show()


# calculate number of unique degrees per member_id:
def calculate_n_degrees(df, id_col):
    rows_per_id = df.groupby(id_col).size().reset_index(name='number of degrees')
    return rows_per_id


# calculate number of unique positions per member_id:
def calculate_n_positions(df, id_col):
    rows_per_id = df.groupby(id_col).size().reset_index(name='number of positions')
    return rows_per_id


def plot_numerical_distribution(df):
    # Select numerical columns
    numerical_columns = df.select_dtypes(include='number').columns

    # Plot each numerical column
    for col in numerical_columns:
        sns.displot(df, x=col, kind="kde", fill=True)
        plt.title(f"Distribution of {col}")
        plt.show()


def plot_pca_2d(df):
    # Fit and transform the data to the PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(df)

    # Create a scatter plot
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Embeddings')
    plt.show()


def plot_pca_3d(df):
    # Fit and transform the data to the PCA
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(df)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2])
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('PCA Visualization of Embeddings')
    plt.show()


# utils helper functions:
def remove_html_tags(df, column):
    df[column] = df[column].apply(lambda x: re.sub('<.*?>', '', x) if pd.notnull(x) else x)
    return df


def plot_missing_data(df):
    # Calculating the percentage of missing values in each column
    missing_data = df.isnull().sum() / len(df) * 100

    # Filter out only columns with missing values
    missing_data = missing_data[missing_data != 0]

    # Sort the missing data in descending order
    missing_data.sort_values(ascending=False, inplace=True)

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(y=missing_data.index, x=missing_data)
    plt.title('Percentage of Missing Data by Feature')
    plt.xlabel('Percentage Missing (%)')
    plt.ylabel('Features')
    plt.show()


def replace_newlines(df, column):
    df[column] = df[column].str.replace(r'\n', ' ', regex=True)
    return df


def remove_string(df, column, string_to_remove):
    df[column] = df[column].str.replace(string_to_remove, ' ')
    return df


def remove_punctuation(df, column):
    df[column] = df[column].str.replace(r'[^\w\s]', '')
    return df


def remove_emojis(df, column):
    df[column] = df[column].apply(lambda x: ''.join(c for c in str(x) if c not in emoji.UNICODE_EMOJI))
    return df


def fill_empty_string_with_nan(df):
    df.replace('', np.nan, inplace=True)
    return df


def fill_empty_string_with_nan(df, columns):
    df[columns] = df[columns].replace('', np.nan)
    return df


def replace_none_with_nan(df, column):
    df[column] = df[column].replace({"none": np.nan})
    return df


def replace_nan_with_NaN(df, column):
    df[column] = df[column].replace('nan', np.nan)
    df[column] = df[column].replace('na', np.nan)
    df[column] = df[column].replace('none', np.nan)
    return df


def count_rows_without_nan(df):
    count = df.notna().all(axis=1).sum()
    return count


def count_nans_per_row(df):
    count = df.isna().sum(axis=1)
    return count


def visualize_normalized_histogram(df, column, top_n=100, figsize=(20, 6)):
    value_counts = df[column].value_counts().nlargest(top_n)
    value_counts_normalized = value_counts / len(df) * 100

    # Generate a color palette
    colors = plt.cm.get_cmap('tab20')(np.arange(top_n))

    plt.figure(figsize=figsize)
    plt.bar(value_counts_normalized.index, value_counts_normalized.values, color=colors)
    plt.xlabel(column)
    plt.ylabel('Percentage')
    plt.title(f'Normalized Value Counts Histogram of {column} (Top {top_n})')
    plt.xticks(rotation=90)
    plt.show()


def visualize_histogram_simple(df, column, top_n=100, figsize=(20, 6)):
    value_counts = df[column].value_counts().nlargest(top_n)

    # Generate a color palette
    colors = plt.cm.get_cmap('tab20')(np.arange(top_n))

    plt.figure(figsize=figsize)
    plt.bar(value_counts.index, value_counts.values, color=colors)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Value Counts Histogram of {column} (Top {top_n})')
    plt.xticks(rotation=90)
    plt.show()


def convert_to_float(df, column):
    try:
        df[column] = df[column].astype(float)
    except ValueError:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def visualize_year_range(df, start_column, end_column):
    start_values = df[start_column].replace([np.inf, -np.inf], np.nan).fillna(-1).astype(int)
    end_values = df[end_column].replace([np.inf, -np.inf], np.nan).fillna(-1).astype(int)

    years = range(start_values.min(), end_values.max() + 1)
    counts = [sum((year >= start_values) & (year <= end_values)) for year in years]

    plt.plot(years, counts)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Year Range Visualization')
    plt.show()


def visualize_histograms(df, column1, column2, bins=10, figsize=(8, 6)):
    plt.figure(figsize=figsize)

    # Generate a color palette
    colors = plt.cm.get_cmap('tab10')(np.arange(2))

    plt.hist(df[column1], bins=bins, alpha=0.7, label=column1, color=colors[0])
    plt.hist(df[column2], bins=bins, alpha=0.7, label=column2, color=colors[1])

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram Comparison')
    plt.legend()
    plt.show()


def visualize_word_cloud(df, column):
    text = ' '.join(df[column].dropna().astype(str).tolist())

    wordcloud = WordCloud(width=800, height=600, background_color='white').generate(text)

    plt.figure(figsize=(12, 9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()


def detect_language(text):
    try:
        return detect(text)
    except:
        return None


def replace_spaces_with_nan(df, column):
    df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
    return df


def remove_tabs(df, column):
    df[column] = df[column].str.replace('\t', '')
    return df


def transform_duration_string(duration_string):
    if isinstance(duration_string, str):

        duration_string = duration_string.strip().lower()  # Convert to lowercase
        if duration_string == "1 year":
            years = 1
            months = 0
            return years, months
        language = detect(duration_string)
        if language == 'en':
            pattern = r'(\d+)(?: year[s]*|y)(?: (\d+)(?: month[s]*|m))?$'
            if re.match(r'less than a year', duration_string):
                return 0, 7
            match = re.search(r'(\d+)(?: year[s]*|y) (\d+)(?: month[s]*|m)$', duration_string)
            if match:
                return int(match.group(1)), int(match.group(2))
            match = re.search(r'(\d+)(?: month[s]*|m)$', duration_string)
            if match:
                return 0, int(match.group(1))
            match = re.search(r'(\d+)(?: year[s]*|y)$', duration_string)
            if match:
                return int(match.group(1)), 0
        elif language == 'es':
            pattern = r'(\d+)(?: año[s]*|a)(?: (\d+)(?: mes[es]*|m))?$'
        elif language == 'ar':
            pattern = r'(\d+)(?: سنة[s]*|س)(?: (\d+)(?: شهر[ا]*|ش))?$'
        elif language == 'fr':
            pattern = r'(\d+)(?: an[s]*|a)(?: (\d+)(?: moi[s]*|m))?$'
        elif language == 'de':
            pattern = r'(\d+)(?: jahr[e]*|j)(?: (\d+)(?: monat[e]*|m))?$'
        elif language == 'it':
            pattern = r'(\d+)(?: anno[i]*|a)(?: (\d+)(?: mes[ei]*|m))?$'
        elif language == 'nl':
            pattern = r'(\d+)(?: jaar[en]*|j)(?: (\d+)(?: maand[en]*|m))?$'
        elif language == 'pl':
            pattern = r'(\d+)(?: rok[ów]*)(?: (\d+)(?: miesiąc[ów]*))?$'
        elif language == 'cs':
            pattern = r'(\d+)(?: rok[ů]*)(?: (\d+)(?: měsíc[ů]*))?$'
        elif language == 'id':
            pattern = r'(\d+)(?: tahun|thn)(?: (\d+)(?: bulan|bln))?$'
        else:
            return np.nan, np.nan

        match = re.search(pattern, duration_string)
        if match:
            years = int(match.group(1) or 0)
            months = int(match.group(2) or 0)
            return years, months
    return np.nan, np.nan


def visualize_top_15_category_histogram(data, category_column, cluster_column, top, title, width, height):
    # Get the top 15 categories by count
    top_n_categories = data[category_column].value_counts().nlargest(top).index.tolist()

    # Filter the data for the top 15 categories
    filtered_data = data[data[category_column].isin(top_n_categories)]

    # Create a histogram of the filtered data with colors based on cluster labels
    fig = px.histogram(filtered_data, x=category_column, color=cluster_column, title=title)

    # Update figure layout
    fig.update_layout(
        autosize=False,
        width=width,
        height=height
    )
    fig.show()


def visualize_three_categorical_variables(data, x_var, y_var, color_var):
    fig = px.bar(data, x=x_var, y=y_var, color=color_var, barmode='group')

    # Update the layout
    fig.update_layout(
        title='Grouped Bar Chart',
        xaxis_title=x_var,
        yaxis_title=y_var,
        legend_title=color_var,
        autosize=False,
        width=2000,
        height=2000
    )

    fig.show()


# alternate function:
# def visualize_top_15_category_histogram(data, category_column, cluster_column, top, title, width, height):
#     # Get the top 15 categories by count
#     top_n_categories = data[category_column].value_counts().nlargest(top).index.tolist()
#
#     # Filter the data for the top 15 categories
#     filtered_data = data[data[category_column].isin(top_n_categories)]
#
#     # Sort the filtered data by category_column
#     filtered_data = filtered_data.sort_values(by=category_column)
#
#     # Create a histogram of the filtered data with colors based on cluster labels
#     fig = px.histogram(filtered_data, x=category_column, color=cluster_column, title=title, histnorm='percent')
#
#     # Update figure layout
#     fig.update_layout(
#         autosize=False,
#         width=width,
#         height=height,
#         yaxis_title='Percentage'
#     )
#     fig.show()


def visualize_pca_3d_with_additional_data(data, title):
    fig = px.scatter_3d(data, x='component 1',
                        y='component 2',
                        z='component 3',
                        color='cluster_scaled_string',
                        title=title
                        )
    fig.update_layout(width=1200, height=800)
    fig.show()


def get_latest_dates(df):
    df['sort_key'] = np.where(df['date_to'] == 0, df['date_from'], df['date_to'])
    df = df.sort_values(['member_id', 'sort_key'], ascending=[True, False])
    latest_dates = df.groupby('member_id').first().reset_index()
    latest_dates = latest_dates.drop(columns=['sort_key'])

    return latest_dates


def transform_experience_dates(experience):
    def transform_date_format(date_value):
        try:
            if isinstance(date_value, int) or date_value.isdigit():
                return str(date_value)  # Return the integer or numeric string as is
            else:
                date_string = str(date_value)
                date_object = datetime.strptime(date_string, "%b-%y")
                return date_object.strftime("%Y-%m")  # Format with year and month only
        except ValueError:
            return None

    def extract_year(value):
        if isinstance(value, str):
            pattern = r'\b(\d{4})\b'  # Regular expression pattern to match a four-digit year
            match = re.search(pattern, value)
            if match:
                return str(match.group(1))
        return None

    experience['transformed_date_from'] = experience['date_from'].apply(transform_date_format)
    experience['transformed_date_to'] = experience['date_to'].apply(transform_date_format)

    experience.loc[experience['transformed_date_from'].isnull(), 'transformed_date_from'] = experience.loc[
        experience['transformed_date_from'].isnull(), 'date_from'].apply(extract_year)
    experience.loc[experience['transformed_date_to'].isnull(), 'transformed_date_to'] = experience.loc[
        experience['transformed_date_to'].isnull(), 'date_to'].apply(extract_year)

    experience['transformed_date_from'] = experience['transformed_date_from'].str.replace(r'-\d{2}$', '', regex=True)
    experience['transformed_date_to'] = experience['transformed_date_to'].str.replace(r'-\d{2}$', '', regex=True)

    return experience


def visualize_none_percentages(df):
    # Convert all values to lowercase strings and check for "none" occurrence
    none_counts = df.select_dtypes(include='object').apply(lambda x: x.str.lower()).eq("none").sum()
    column_percentages = none_counts / len(df) * 100

    # Create a bar plot to visualize the percentages
    column_percentages.plot(kind='bar')
    plt.xlabel('Columns')
    plt.ylabel('Percentage of "none"')
    plt.title('Percentage of "none" in String Columns')
    plt.show()
