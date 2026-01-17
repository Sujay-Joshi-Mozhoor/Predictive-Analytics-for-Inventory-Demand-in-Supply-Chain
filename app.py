import pandas as pd
import numpy as np
import panel as pn
import hvplot.pandas  # noqa
import holoviews as hv
import matplotlib.pyplot as plt

# =========================================================
# Extensions
# =========================================================
pn.extension('tabulator')
hv.extension('bokeh')
hv.renderer('bokeh').shared_axes = False

# =========================================================
# Global CSS
# =========================================================
pn.config.raw_css.append("""
body {
    background-image: url("https://images.unsplash.com/photo-1581092334494-1c0e7b8f8b5b");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
.bk-root .pn-template {
    background: rgba(255, 255, 255, 0.92) !important;
}
.pn-card {
    background: rgba(255, 255, 255, 0.97) !important;
}
""")

# =========================================================
# Load data (RELATIVE PATHS ONLY)
# =========================================================
final_timeseries_df = pd.read_csv("final_timeseries_df.csv")
final_timeseries_df['date'] = pd.to_datetime(final_timeseries_df['date'])

full_data_final = pd.read_csv("full_data_final.csv")
full_data_final['order_date'] = pd.to_datetime(full_data_final['order_date'])

df = final_timeseries_df.copy()

# =========================================================
# UI constants
# =========================================================
PLOT_WIDTH = 720
PLOT_HEIGHT = 360
CARD_WIDTH = 800
WIDGET_WIDTH = 320
WIDGET_BLOCK_HEIGHT = 120

# =========================================================
# Widgets
# =========================================================
category_selector = pn.widgets.Select(
    name="Category",
    options=sorted(df['category'].unique()),
    width=WIDGET_WIDTH
)

campaign_selector = pn.widgets.Select(
    name="Campaign",
    options={
        "No Campaign (0%)": 0.00,
        "Low Campaign (+5%)": 0.05,
        "Moderate Campaign (+10%)": 0.10,
        "Intense Campaign (+20%)": 0.20
    },
    width=WIDGET_WIDTH
)

# =========================================================
# Widget blocks
# =========================================================
widgets_chart1 = pn.Column(category_selector, height=WIDGET_BLOCK_HEIGHT)
widgets_chart2 = pn.Column(category_selector, campaign_selector, height=WIDGET_BLOCK_HEIGHT)

alert_block = pn.Column(
    pn.pane.Markdown("<b style='color:red'>⚠ Model accuracy has dropped (MAPE = 35%)</b>"),
    height=WIDGET_BLOCK_HEIGHT
)

empty_block = pn.Column(pn.Spacer(height=WIDGET_BLOCK_HEIGHT))

# =========================================================
# Chart 1
# =========================================================
@pn.depends(category_selector)
def chart1(category):
    dff = df[df['category'] == category]
    return dff.hvplot.line(
        x='date', y='orders', by='flag',
        width=PLOT_WIDTH, height=PLOT_HEIGHT,
        linewidth=2, legend='top_left',
        title=f"Weekly Orders – {category}"
    )

# =========================================================
# Chart 2
# =========================================================
@pn.depends(category_selector, campaign_selector)
def chart2(category, uplift):
    dff = df[df['category'] == category]
    train_df = dff[dff['flag'] == 'train']
    forecast_df = dff[dff['flag'] == 'forecast'].copy()

    train_plot = train_df.hvplot.line(x='date', y='orders', label='Actuals')

    if forecast_df.empty:
        return train_plot

    forecast_df['campaign_orders'] = forecast_df['orders'] * (1 + uplift)

    return (
        train_plot *
        forecast_df.hvplot.line(
            x='date', y='campaign_orders',
            label=f'Campaign Forecast (+{int(uplift*100)}%)'
        )
    ).opts(width=PLOT_WIDTH, height=PLOT_HEIGHT)

# =========================================================
# Chart 3 (Mock Accuracy)
# =========================================================
def chart3():
    dates = pd.date_range(start='2018-01-01', periods=8, freq='W')
    return pd.DataFrame({
        'date': dates,
        'orders': np.random.randint(80, 120, size=len(dates))
    }).hvplot.line(x='date', y='orders', width=PLOT_WIDTH, height=PLOT_HEIGHT)

# =========================================================
# Chart 4
# =========================================================
def chart4():
    weekly_cat = full_data_final.groupby(
        ['product_category_name',
         pd.Grouper(key='order_date', freq='W-SUN')]
    ).size().reset_index(name='qty')

    top5 = (
        weekly_cat.groupby('product_category_name')['qty']
        .sum().sort_values(ascending=False).head(5)
    )

    plt.figure(figsize=(6, 4))
    plt.bar(top5.index, top5.values)
    plt.title("Top 5 Product Categories")
    plt.xticks(rotation=45)
    plt.tight_layout()

    pane = pn.pane.Matplotlib(plt.gcf(), width=PLOT_WIDTH, height=PLOT_HEIGHT)
    plt.close()
    return pane

# =========================================================
# Layout
# =========================================================
row1 = pn.Row(
    pn.Card(pn.Column(widgets_chart1, chart1), title="Weekly Orders", width=CARD_WIDTH),
    pn.Spacer(width=40),
    pn.Card(pn.Column(widgets_chart2, chart2), title="Campaign Impact", width=CARD_WIDTH),
)

row2 = pn.Row(
    pn.Card(pn.Column(alert_block, chart3), title="Forecast Accuracy", width=CARD_WIDTH),
    pn.Spacer(width=40),
    pn.Card(pn.Column(empty_block, chart4), title="Top Categories", width=CARD_WIDTH),
)

template = pn.template.FastListTemplate(
    title="📈 Predictive Analytics for Inventory Demand in Supply Chain",
    main=[row1, pn.Spacer(height=30), row2]
)

template.servable()
