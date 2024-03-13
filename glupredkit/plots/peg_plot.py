import matplotlib.pyplot as plt
import itertools
import os
import sys
import numpy as np
from datetime import datetime
from .base_plot import BasePlot
from shapely.geometry import Point, Polygon
from glupredkit.helpers.unit_config_manager import unit_config_manager


class Plot(BasePlot):
    def __init__(self):
        super().__init__()    

    def __call__(self, models_data):
        if unit_config_manager.use_mgdl:
            unit = "mg/dL"
            max_val = 400
        else:
            unit = "mmol/L"
            max_val = unit_config_manager.convert_value(400)

        plt.figure(figsize=(10, 8))

        for model_data in models_data:
            if unit_config_manager.use_mgdl:
                y_pred = model_data.get('y_pred')
                y_true = model_data.get('y_true')
            else:
                y_pred = [unit_config_manager.convert_value(val) for val in model_data.get('y_pred')]
                y_true = [unit_config_manager.convert_value(val) for val in model_data.get('y_true')]

        ax = parkes(1, y_true, y_pred, 'mgdl')
        print(ax)

        file_path = "data/figures/"
        os.makedirs(file_path, exist_ok=True)
        config_str = sys.argv[3].replace(".pkl", "")
        print(sys.argv[3])
        file_name = f'Parkes_Error_Grid_{config_str}.png'
        plt.savefig(file_path + file_name)
        plt.show()



'''
Code below is from the methcomp open source library: https://github.com/wptmdoorn/methcomp/blob/master/methcomp/glucose.py

MIT License

Copyright (c) 2019-2020, William P.T.M. van Doorn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
class _Parkes(object):
    """Internal class for drawing a Parkes consensus error grid plot"""

    def __init__(
        self,
        type,
        reference,
        test,
        units,
        x_title,
        y_title,
        graph_title,
        xlim,
        ylim,
        color_grid,
        color_gridlabels,
        color_points,
        grid,
        percentage,
        point_kws,
        grid_kws,
    ):
        # variables assignment
        self.type: int = type
        self.reference: np.array = np.asarray(reference)
        self.test: np.array = np.asarray(test)
        self.units = units
        self.graph_title: str = graph_title
        self.x_title: str = x_title
        self.y_title: str = y_title
        self.xlim: list = xlim
        self.ylim: list = ylim
        self.color_grid: str = color_grid
        self.color_gridlabels: str = color_gridlabels
        self.color_points: str = color_points
        self.grid: bool = grid
        self.percentage: bool = percentage
        self.point_kws = {} if point_kws is None else point_kws.copy()
        self.grid_kws = {} if grid_kws is None else grid_kws.copy()

        self._check_params()
        self._derive_params()

    def _check_params(self):
        if self.type != 1 and self.type != 2:
            raise ValueError("Type of Diabetes should either be 1 or 2.")

        if len(self.reference) != len(self.test):
            raise ValueError("Length of reference and test values are not equal")

        if self.units not in ["mmol", "mg/dl", "mgdl"]:
            raise ValueError(
                "The provided units should be one of the following:"
                " mmol, mgdl or mg/dl."
            )

        if any(
            [
                x is not None and not isinstance(x, str)
                for x in [self.x_title, self.y_title]
            ]
        ):
            raise ValueError("Axes labels arguments should be provided as a str.")

    def _derive_params(self):
        if self.x_title is None:
            _unit = "mmol/L" if self.units == "mmol" else "mg/dL"
            self.x_title = "Reference glucose concentration ({})".format(_unit)

        if self.y_title is None:
            _unit = "mmol/L" if self.units == "mmol" else "mg/dL"
            self.y_title = "Predicted glucose concentration ({})".format(_unit)

    def _coef(self, x, y, xend, yend):
        if xend == x:
            raise ValueError("Vertical line - function inapplicable")
        return (yend - y) / (xend - x)

    def _endy(self, startx, starty, maxx, coef):
        return (maxx - startx) * coef + starty

    def _endx(self, startx, starty, maxy, coef):
        return (maxy - starty) / coef + startx

    def _calc_error_zone(self):
        # ref, pred
        ref = self.reference
        pred = self.test

        # calculate conversion factor if needed
        n = 18 if self.units == "mmol" else 1

        maxX = max(max(ref) + 20 / n, 550 / n)
        maxY = max([*(np.array(pred) + 20 / n), maxX, 550 / n])

        # we initialize an array with ones
        # this in fact very smart because all the non-matching values will automatically
        # end up in zone A (which is zero)
        _zones = np.zeros(len(ref))

        if self.type == 1:
            ce = self._coef(35, 155, 50, 550)
            cdu = self._coef(80, 215, 125, 550)
            cdl = self._coef(250, 40, 550, 150)
            ccu = self._coef(70, 110, 260, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(280, 380, 430, 550)
            cbl = self._coef(385, 300, 550, 450)

            limitE1 = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [0, 35 / n, self._endx(35 / n, 155 / n, maxY, ce), 0, 0],
                        [150 / n, 155 / n, maxY, maxY, 150 / n],
                    )
                ]
            )

            limitD1L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [250 / n, 250 / n, maxX, maxX, 250 / n],
                        [0, 40 / n, self._endy(410 / n, 110 / n, maxX, cdl), 0, 0],
                    )
                ]
            )

            limitD1U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            25 / n,
                            50 / n,
                            80 / n,
                            self._endx(80 / n, 215 / n, maxY, cdu),
                            0,
                            0,
                        ],
                        [100 / n, 100 / n, 125 / n, 215 / n, maxY, maxY, 100 / n],
                    )
                ]
            )

            limitC1L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [120 / n, 120 / n, 260 / n, maxX, maxX, 120 / n],
                        [
                            0,
                            30 / n,
                            130 / n,
                            self._endy(260 / n, 130 / n, maxX, ccl),
                            0,
                            0,
                        ],
                    )
                ]
            )

            limitC1U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            30 / n,
                            50 / n,
                            70 / n,
                            self._endx(70 / n, 110 / n, maxY, ccu),
                            0,
                            0,
                        ],
                        [60 / n, 60 / n, 80 / n, 110 / n, maxY, maxY, 60 / n],
                    )
                ]
            )

            limitB1L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [50 / n, 50 / n, 170 / n, 385 / n, maxX, maxX, 50 / n],
                        [
                            0,
                            30 / n,
                            145 / n,
                            300 / n,
                            self._endy(385 / n, 300 / n, maxX, cbl),
                            0,
                            0,
                        ],
                    )
                ]
            )

            limitB1U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            30 / n,
                            140 / n,
                            280 / n,
                            self._endx(280 / n, 380 / n, maxY, cbu),
                            0,
                            0,
                        ],
                        [50 / n, 50 / n, 170 / n, 380 / n, maxY, maxY, 50 / n],
                    )
                ]
            )

            for i, points in enumerate(zip(ref, pred)):
                for f, r in zip(
                    [
                        limitB1L,
                        limitB1U,
                        limitC1L,
                        limitC1U,
                        limitD1L,
                        limitD1U,
                        limitE1,
                    ],
                    [1, 1, 2, 2, 3, 3, 4],
                ):
                    if f.contains(Point(points[0], points[1])):
                        _zones[i] = r

            return [int(i) for i in _zones]

        elif self.type == 2:
            ce = self._coef(35, 200, 50, 550)
            cdu = self._coef(35, 90, 125, 550)
            cdl = self._coef(410, 110, 550, 160)
            ccu = self._coef(30, 60, 280, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(230, 330, 440, 550)
            cbl = self._coef(330, 230, 550, 450)

            limitE2 = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            35 / n,
                            self._endx(35 / n, 200 / n, maxY, ce),
                            0,
                            0,
                        ],  # x limits E upper
                        [200 / n, 200 / n, maxY, maxY, 200 / n],
                    )
                ]
            )  # y limits E upper

            limitD2L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            250 / n,
                            250 / n,
                            410 / n,
                            maxX,
                            maxX,
                            250 / n,
                        ],  # x limits D lower
                        [
                            0,
                            40 / n,
                            110 / n,
                            self._endy(410 / n, 110 / n, maxX, cdl),
                            0,
                            0,
                        ],
                    )
                ]
            )  # y limits D lower

            limitD2U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            25 / n,
                            35 / n,
                            self._endx(35 / n, 90 / n, maxY, cdu),
                            0,
                            0,
                        ],  # x limits D upper
                        [80 / n, 80 / n, 90 / n, maxY, maxY, 80 / n],
                    )
                ]
            )  # y limits D upper

            limitC2L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [90 / n, 260 / n, maxX, maxX, 90 / n],  # x limits C lower
                        [0, 130 / n, self._endy(260 / n, 130 / n, maxX, ccl), 0, 0],
                    )
                ]
            )  # y limits C lower

            limitC2U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            30 / n,
                            self._endx(30 / n, 60 / n, maxY, ccu),
                            0,
                            0,
                        ],  # x limits C upper
                        [60 / n, 60 / n, maxY, maxY, 60 / n],
                    )
                ]
            )  # y limits C upper

            limitB2L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            50 / n,
                            50 / n,
                            90 / n,
                            330 / n,
                            maxX,
                            maxX,
                            50 / n,
                        ],  # x limits B lower
                        [
                            0,
                            30 / n,
                            80 / n,
                            230 / n,
                            self._endy(330 / n, 230 / n, maxX, cbl),
                            0,
                            0,
                        ],
                    )
                ]
            )  # y limits B lower

            limitB2U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            30 / n,
                            230 / n,
                            self._endx(230 / n, 330 / n, maxY, cbu),
                            0,
                            0,
                        ],  # x limits B upper
                        [50 / n, 50 / n, 330 / n, maxY, maxY, 50 / n],
                    )
                ]
            )  # y limits B upper

            for i, points in enumerate(zip(ref, pred)):
                for f, r in zip(
                    [
                        limitB2L,
                        limitB2U,
                        limitC2L,
                        limitC2U,
                        limitD2L,
                        limitD2U,
                        limitE2,
                    ],
                    [1, 1, 2, 2, 3, 3, 4],
                ):
                    if f.contains(Point(points[0], points[1])):
                        _zones[i] = r

            return [int(i) for i in _zones]

    def plot(self, ax):
        # ref, pred
        ref = self.reference
        pred = self.test

        # calculate conversion factor if needed
        n = 18 if self.units == "mmol" else 1

        maxX = self.xlim or max(max(ref) + 20 / n, 550 / n)
        maxY = self.ylim or max([*(np.array(pred) + 20 / n), maxX, 550 / n])

        if self.type == 1:
            ce = self._coef(35, 155, 50, 550)
            cdu = self._coef(80, 215, 125, 550)
            cdl = self._coef(250, 40, 550, 150)
            ccu = self._coef(70, 110, 260, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(280, 380, 430, 550)
            cbl = self._coef(385, 300, 550, 450)

            _gridlines = [
                ([0, min(maxX, maxY)], [0, min(maxX, maxY)], ":"),
                ([0, 30 / n], [50 / n, 50 / n], "-"),
                ([30 / n, 140 / n], [50 / n, 170 / n], "-"),
                ([140 / n, 280 / n], [170 / n, 380 / n], "-"),
                (
                    [280 / n, self._endx(280 / n, 380 / n, maxY, cbu)],
                    [380 / n, maxY],
                    "-",
                ),
                ([50 / n, 50 / n], [0 / n, 30 / n], "-"),
                ([50 / n, 170 / n], [30 / n, 145 / n], "-"),
                ([170 / n, 385 / n], [145 / n, 300 / n], "-"),
                (
                    [385 / n, maxX],
                    [300 / n, self._endy(385 / n, 300 / n, maxX, cbl)],
                    "-",
                ),
                ([0 / n, 30 / n], [60 / n, 60 / n], "-"),
                ([30 / n, 50 / n], [60 / n, 80 / n], "-"),
                ([50 / n, 70 / n], [80 / n, 110 / n], "-"),
                (
                    [70 / n, self._endx(70 / n, 110 / n, maxY, ccu)],
                    [110 / n, maxY],
                    "-",
                ),
                ([120 / n, 120 / n], [0 / n, 30 / n], "-"),
                ([120 / n, 260 / n], [30 / n, 130 / n], "-"),
                (
                    [260 / n, maxX],
                    [130 / n, self._endy(260 / n, 130 / n, maxX, ccl)],
                    "-",
                ),
                ([0 / n, 25 / n], [100 / n, 100 / n], "-"),
                ([25 / n, 50 / n], [100 / n, 125 / n], "-"),
                ([50 / n, 80 / n], [125 / n, 215 / n], "-"),
                (
                    [80 / n, self._endx(80 / n, 215 / n, maxY, cdu)],
                    [215 / n, maxY],
                    "-",
                ),
                ([250 / n, 250 / n], [0 / n, 40 / n], "-"),
                (
                    [250 / n, maxX],
                    [40 / n, self._endy(410 / n, 110 / n, maxX, cdl)],
                    "-",
                ),
                ([0 / n, 35 / n], [150 / n, 155 / n], "-"),
                ([35 / n, self._endx(35 / n, 155 / n, maxY, ce)], [155 / n, maxY], "-"),
            ]

        elif self.type == 2:
            ce = self._coef(35, 200, 50, 550)
            cdu = self._coef(35, 90, 125, 550)
            cdl = self._coef(410, 110, 550, 160)
            ccu = self._coef(30, 60, 280, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(230, 330, 440, 550)
            cbl = self._coef(330, 230, 550, 450)

            _gridlines = [
                ([0, min(maxX, maxY)], [0, min(maxX, maxY)], ":"),
                ([0, 30 / n], [50 / n, 50 / n], "-"),
                ([30 / n, 230 / n], [50 / n, 330 / n], "-"),
                (
                    [230 / n, self._endx(230 / n, 330 / n, maxY, cbu)],
                    [330 / n, maxY],
                    "-",
                ),
                ([50 / n, 50 / n], [0 / n, 30 / n], "-"),
                ([50 / n, 90 / n], [30 / n, 80 / n], "-"),
                ([90 / n, 330 / n], [80 / n, 230 / n], "-"),
                (
                    [330 / n, maxX],
                    [230 / n, self._endy(330 / n, 230 / n, maxX, cbl)],
                    "-",
                ),
                ([0 / n, 30 / n], [60 / n, 60 / n], "-"),
                ([30 / n, self._endx(30 / n, 60 / n, maxY, ccu)], [60 / n, maxY], "-"),
                ([90 / n, 260 / n], [0 / n, 130 / n], "-"),
                (
                    [260 / n, maxX],
                    [130 / n, self._endy(260 / n, 130 / n, maxX, ccl)],
                    "-",
                ),
                ([0 / n, 25 / n], [80 / n, 80 / n], "-"),
                ([25 / n, 35 / n], [80 / n, 90 / n], "-"),
                ([35 / n, self._endx(35 / n, 90 / n, maxY, cdu)], [90 / n, maxY], "-"),
                ([250 / n, 250 / n], [0 / n, 40 / n], "-"),
                ([250 / n, 410 / n], [40 / n, 110 / n], "-"),
                (
                    [410 / n, maxX],
                    [110 / n, self._endy(410 / n, 110 / n, maxX, cdl)],
                    "-",
                ),
                ([0 / n, 35 / n], [200 / n, 200 / n], "-"),
                ([35 / n, self._endx(35 / n, 200 / n, maxY, ce)], [200 / n, maxY], "-"),
            ]

        colors = ["#196600", "#7FFF00", "#FF7B00", "#FF5700", "#FF0000"]

        _gridlabels = [
            (500, 500, "A", colors[0]),
            (300, 500, "B", colors[1]),
            (500, 320, "B", colors[1]),
            (165, 500, "C", colors[2]),
            (500, 190, "C", colors[2]),
            (500, 50, "D", colors[3]),
            (75, 500, "D", colors[3]),
            (15, 500, "E", colors[4]),
        ]

        # plot individual points
        if self.color_points == "auto":
            ax.scatter(
                self.reference,
                self.test,
                marker="o",
                alpha=0.6,
                c=[colors[i] for i in self._calc_error_zone()],
                s=8,
                **self.point_kws
            )
        else:
            ax.scatter(
                self.reference,
                self.test,
                marker="o",
                color=self.color_points,
                alpha=0.6,
                s=8,
                **self.point_kws
            )

        # plot grid lines
        if self.grid:
            for g in _gridlines:
                ax.plot(
                    np.array(g[0]),
                    np.array(g[1]),
                    g[2],
                    color=self.color_grid,
                    **self.grid_kws
                )

            if self.percentage:
                zones = [["A", "B", "C", "D", "E"][i] for i in self._calc_error_zone()]

                for label in _gridlabels:
                    ax.text(
                        label[0] / n,
                        label[1] / n,
                        label[2],
                        fontsize=12,
                        fontweight="bold",
                        color=label[3]
                        if self.color_gridlabels == "auto"
                        else self.color_gridlabels,
                    )
                    ax.text(
                        label[0] / n + (18 / n),
                        label[1] / n + (18 / n),
                        "{:.1f}".format((zones.count(label[2]) / len(zones)) * 100),
                        fontsize=9,
                        fontweight="bold",
                        color=label[3]
                        if self.color_gridlabels == "auto"
                        else self.color_gridlabels,
                    )

            else:
                for label in _gridlabels:
                    ax.text(
                        label[0] / n,
                        label[1] / n,
                        label[2],
                        fontsize=12,
                        fontweight="bold",
                        color=label[3]
                        if self.color_gridlabels == "auto"
                        else self.color_gridlabels,
                    )

        # limits and ticks
        _ticks = [
            70,
            100,
            150,
            180,
            240,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            800,
            850,
            900,
            950,
            1000,
        ]

        ax.set_xticks([round(x / n, 1) for x in _ticks])
        ax.set_yticks([round(x / n, 1) for x in _ticks])
        ax.set_xlim(0, maxX)
        ax.set_ylim(0, maxY)

        # graph labels
        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        if self.graph_title is not None:
            ax.set_title(self.graph_title)


def parkes(
    type,
    reference,
    test,
    units,
    x_label=None,
    y_label=None,
    title=None,
    xlim=None,
    ylim=None,
    color_grid="#000000",
    color_gridlabels="auto",
    color_points="auto",
    grid=True,
    percentage=False,
    point_kws=None,
    grid_kws=None,
    square=False,
    ax=None,
):
    """Provide a glucose error grid analyses as designed by Parkes.

    This is an Axis-level function which will draw the Parke-error grid plot.
    onto the current active Axis object unless ``ax`` is provided.


    Parameters
    ----------
    type : int
        Parkes error grid differ for each type of diabetes. This should be either
        1 or 2 corresponding to the type of diabetes.
    reference, test : array, or list
        Glucose values obtained from the reference and predicted methods, preferably
        provided in a np.array.
    units : str
        The SI units which the glucose values are provided in.
        Options: 'mmol', 'mgdl' or 'mg/dl'.
    x_label : str, optional
        The label which is added to the X-axis. If None is provided, a standard
        label will be added.
    y_label : str, optional
        The label which is added to the Y-axis. If None is provided, a standard
        label will be added.
    title : str, optional
        Title of the Parkes-error grid plot. If None is provided, no title will be
        plotted.
    xlim : list, optional
        Minimum and maximum limits for X-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    ylim : list, optional
        Minimum and maximum limits for Y-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    color_grid : str, optional
        Color of the Clarke error grid lines. Defaults to #000000 which represents
        the black color.
    color_gridlabels : str, optional
        Color of the grid labels (A, B, C, ..) that will be plotted.
        Defaults to 'auto' which colors the points according to their relative zones.
    color_points : str, optional
        Color of the individual differences that will be plotted. Defaults to 'auto'
        which colors the points according to their relative zones.
    grid : bool, optional
        Enable the grid lines of the Parkes error. Defaults to True.
    percentage : bool, optional
        If True, percentage of the zones will be depicted in the plot.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be square-shaped.
    point_kws : dict of key, value mappings, optional
        Additional keyword arguments for `plt.scatter`.
    grid_kws : dict of key, value mappings, optional
        Additional keyword arguments for the grid with `plt.plot`.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the Parkes error grid plot.

    References
    ----------
    [parkes_2000] Parkes, J. L., Slatin S. L. et al.
                  Diabetes Care, vol. 23, no. 8, 2000, pp. 1143-1148.
    [pfutzner_2013] Pfutzner, A., Klonoff D. C., et al.
                    J Diabetes Sci Technol, vol. 7, no. 5, 2013, pp. 1275-1281.
    """

    plotter: _Parkes = _Parkes(
        type,
        reference,
        test,
        units,
        x_label,
        y_label,
        title,
        xlim,
        ylim,
        color_grid,
        color_gridlabels,
        color_points,
        grid,
        percentage,
        point_kws,
        grid_kws,
    )

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    if square:
        ax.set_aspect("equal")

    plotter.plot(ax)

    return ax
