import os
from yaml import load
from statistics import mean

results = dict(reads={},
               trimmed={})

for f in os.listdir():
    if not f.endswith(".yml"):
        continue
    result = load(open(f))
    results[f.split("-")[0]][int(f.split("-")[1][0])] = dict(accuracy=mean([x["accuracy"] for x in result]),
                                                             precision=mean([x["precision"] for x in result]),
                                                             recall=mean([x["recall"] for x in result]),
                                                             train_duration_sec=mean([x["train_duration_sec"] for x in result]))

k = [3, 4, 5, 6, 7]

from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Set2_3
from bokeh.io import export_svgs


# for the SG data
p = figure(title="SG", x_axis_label='k', y_axis_label='mean accuracy %', plot_width=600, plot_height=450)
p.line(k, [results["reads"][_k]["accuracy"] * 100 for _k in k], legend="Logistic Regression", line_color=Set2_3[0], line_width=3)
p.line(k, [17.02, 32.98, 59.80, 80.77, 85.50], legend="CNN", line_dash="dashed", line_color=Set2_3[1], line_width=3)
p.line(k, [17.75, 54.11, 71.44, 77.85, 81.27], legend="DBN", line_dash="dotted", line_color=Set2_3[2], line_width=3)
p.legend.location = "bottom_right"
p.xaxis.ticker = k
p.yaxis.ticker = list(range(0, 101, 10))
p.output_backend = "svg"
export_svgs(p, filename="sg.svg")

# for the AMP data
p = figure(title="AMP", x_axis_label='k', y_axis_label='mean accuracy %', plot_width=600, plot_height=450)
p.line(k, [results["trimmed"][_k]["accuracy"] * 100 for _k in k], legend="Logistic Regression", line_color=Set2_3[0], line_width=3)
p.line(k, [51.01, 77.69, 88.13, 90.92, 91.33], legend="CNN", line_dash="dashed", line_color=Set2_3[1], line_width=3)
p.line(k, [56.69, 85.10, 89.82, 90.55, 91.37], legend="DBN", line_dash="dotted", line_color=Set2_3[2], line_width=3)
p.legend.location = "bottom_right"
p.xaxis.ticker = k
p.yaxis.ticker = list(range(0, 101, 10))
p.output_backend = "svg"
export_svgs(p, filename="amp.svg")


# for the time data
p = figure(title="", x_axis_label='k', y_axis_label='mean training time (s)', plot_width=600, plot_height=450)
p.line(k,
       [mean((results["reads"][_k]["train_duration_sec"], results["trimmed"][_k]["train_duration_sec"])) for _k in k],
       legend="Logistic Regression",
       line_color=Set2_3[0],
       line_width=3)

p.line(k, [686.403, 1256.652, 3091.721, 8021.737, 24204.754], line_dash="dashed", legend="CNN", line_color=Set2_3[1], line_width=3)
p.line(k, [7288.913, 8170.077, 11875.716, 20346.112, 37161.237], line_dash="dotted", legend="DBN", line_color=Set2_3[2], line_width=3)
p.legend.location = "top_left"
p.xaxis.ticker = k
p.output_backend = "svg"
export_svgs(p, filename="time.svg")
