from dash import Dash, html, dcc, Input, Output, ctx, callback
import dash_cytoscape as cyto

cyto.load_extra_layouts()

app = Dash(__name__)

nodes = [
    {
        'data': {'id': short, 'label': label, 'classes': class_},
    }

    # S̅F̅ T̅S̅ M̅E̅D̅
    # S̲F̲ T̲S̲ M̲E̲D̲

    for short, label, class_ in (
        ('notSF_notTS_notMED', 'S̲F̲ T̲S̲ M̲E̲D̲', 'label_text'),
        ('SF_notTS_notMED', 'S̅F̅ T̲S̲ M̲E̲D̲', 'label_text'),
        ('SF_TS_notMED', 'S̅F̅ T̅S̅ M̲E̲D̲', 'label_text'),
        ('SF_TS_MED', 'S̅F̅ T̅S̅ M̅E̅D̅', 'label_text'),
        ('SF_notTS_MED', 'S̅F̅ T̲S̲ M̅E̅D̅', 'label_text'),
        ('notSF_notTS_MED', 'S̲F̲ T̲S̲ M̅E̅D̅', 'label_text'),
    )
]

edges = [
    {'data': {'source': source, 'target': target, 'classes': class_}}
    for source, target, class_ in (
        ('notSF_notTS_notMED', 'SF_notTS_notMED', 'edge_notSF_notTs_notMED'),
        ('notSF_notTS_notMED', 'SF_notTS_MED', 'edge_notSF_notTs_notMED'),
        ('notSF_notTS_notMED', 'notSF_notTS_MED', 'edge_notSF_notTs_notMED'),

        # ('SF_notTS_notMED', 'notSF_notTS_notMED', 'SF_notTS_notMED'),
        ('SF_notTS_notMED', 'SF_TS_notMED', 'SF_notTS_notMED'),
        ('SF_notTS_notMED', 'SF_TS_MED', 'SF_notTS_notMED'),
        ('SF_notTS_notMED', 'SF_notTS_MED', 'SF_notTS_notMED'),
        ('SF_notTS_notMED', 'notSF_notTS_MED', 'SF_notTS_notMED'),

        ('SF_TS_notMED', 'notSF_notTS_notMED', 'SF_TS_notMED'),
        ('SF_TS_notMED', 'SF_notTS_notMED', 'SF_TS_notMED'),
        ('SF_TS_notMED', 'SF_TS_MED', 'SF_TS_notMED'),
        ('SF_TS_notMED', 'SF_notTS_MED', 'SF_TS_notMED'),
        ('SF_TS_notMED', 'notSF_notTS_MED', 'SF_TS_notMED'),

        ('SF_TS_MED', 'notSF_notTS_notMED', 'SF_TS_MED'),
        ('SF_TS_MED', 'SF_notTS_notMED', 'SF_TS_MED'),
        ('SF_TS_MED', 'SF_TS_notMED', 'SF_TS_MED'),
        ('SF_TS_MED', 'SF_notTS_MED', 'SF_TS_MED'),
        ('SF_TS_MED', 'notSF_notTS_MED', 'SF_TS_MED'),

        ('SF_notTS_MED', 'notSF_notTS_notMED', 'SF_notTS_MED'),
        ('SF_notTS_MED', 'SF_notTS_notMED', 'SF_notTS_MED'),
        ('SF_notTS_MED', 'SF_TS_notMED', 'SF_notTS_MED'),
        ('SF_notTS_MED', 'SF_TS_MED', 'SF_notTS_MED'),
        ('SF_notTS_MED', 'notSF_notTS_MED', 'SF_notTS_MED'),

        ('notSF_notTS_MED', 'notSF_notTS_notMED', 'notSF_notTS_MED'),
        ('notSF_notTS_MED', 'SF_notTS_notMED', 'notSF_notTS_MED'),
        ('notSF_notTS_MED', 'SF_notTS_MED', 'notSF_notTS_MED'),
    )
]

elements = nodes + edges

app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape-layout-2',
        elements=elements,
        style={'width': '100%', 'height': '100vh'},
        layout={
            'name': 'circle'
        },
        responsive=True,
        stylesheet = [
            {
                'selector': 'label_text',
                'style': {
                    'label': 'data(label)',
                    'text-wrap': 'wrap',
                    'text-max-width': '200px',
                    # 'text-valign': 'bottom',
                    'text-halign': 'center',
                    'font-size': '20px',
                    'font-family': 'Arial, sans-serif',
                    'color': 'black',
                    'text-outline-color': 'white',
                    'text-outline-width': '2px',
                    # 'text-decoration': 'overline',
                    # 'font-decoration': 'overline'
                },
            },
            {
                    'selector': 'edge',
                    'style': {
                    "target-arrow-shape": "triangle",
                    # "source-arrow-shape": "triangle",
                    'curve-style': 'bezier',
                    'arrow-scale': 2,
                }
            },

            # {
            #     "selector": ":selected",
            #     "style": {"border-width": 3, "border-color": "#333"},
            # },

        ]

    ),
    html.Div(className='four columns', children=[
            html.Div('Download graph:'),
            html.Button("as jpg", id="btn-get-jpg"),
            html.Button("as png", id="btn-get-png"),
            html.Button("as svg", id="btn-get-svg")
        ])
])

@callback(
    Output("cytoscape-layout-2", "generateImage"),
    [
        Input("btn-get-jpg", "n_clicks"),
        Input("btn-get-png", "n_clicks"),
        Input("btn-get-svg", "n_clicks"),
    ])
def get_image(get_jpg_clicks, get_png_clicks, get_svg_clicks):

    # File type to output of 'svg, 'png', 'jpg', or 'jpeg' (alias of 'jpg')
    ftype = 'svg' # default

    # 'store': Stores the image data in 'imageData' !only jpg/png are supported
    # 'download'`: Downloads the image as a file with all data handling
    # 'both'`: Stores image data and downloads image as file.
    action = "download"

    if ctx.triggered:
        if ctx.triggered_id == "btn-get-jpg":
            ftype = 'jpg'
        elif ctx.triggered_id == "btn-get-png":
            ftype = 'png'
        elif ctx.triggered_id == "btn-get-svg":
            ftype = 'svg'

    return {
        'type': ftype,
        'action': action
        }


if __name__ == '__main__':
    app.run(debug=True)
