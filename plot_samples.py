import numpy as np
import plotly.graph_objects as go
from config import get_args
from dataset import Batch_Dataset
from plotly.subplots import make_subplots
import wandb
import plotly.io as pio

args = get_args()
data = Batch_Dataset(args.dataset)
wandb.init(project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name="Plot_samples")

samples = {label: None for label in range(10)}

# Iterate through dataset to find one sample for each class
for i in range(len(data.x_train)):
    label = np.argmax(data.y_train[i])
    img = data.x_train[i]
    if samples[label] is None:
        samples[label] = img
    if all(v is not None for v in samples.values()):  # Stop when all classes are found
        break





subplot_titles = [data.classes[label] for (label,_) in samples.items()]
fig = make_subplots(rows=2, cols=5, vertical_spacing=0.1, horizontal_spacing=0.02,subplot_titles=subplot_titles)

for i, (label,img) in enumerate(samples.items()):
    row = i // 5 + 1
    col = i % 5 + 1
    fig.add_trace(go.Heatmap(z=np.flip(np.reshape(img,(28,28)),axis=0), colorscale='gray', showscale=False),col=col,row=row)

# Update layout: adjust overall figure size and hide axes.
fig.update_layout(
    height=600,
    width=1000,
    title_text="Sample Images from Each Fashion MNIST Class",
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=50)
)

wandb.log({"Samples": wandb.Html(pio.to_html(fig,full_html=False))})

wandb.finish()