import ANNarchy as ann
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Iterable


def ceil(a: float, precision: int = 0):
    """
    Calculate the ceiling value of a number 'a' with a specified precision.

    Parameters:
    :param a: The number for which the ceiling value needs to be calculated.
    :param precision: The number of decimal places to consider for precision (default is 0).

    :return: The ceiling value of 'a' with the specified precision.

    Example:
    ceil(3.14159, 2) returns 3.15
    ceil(5.67, 0) returns 6.0
    """
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def find_largest_factors(c: int):
    """
    Returns the two largest factors a and b of an integer c, such that a * b = c.
    """
    for a in range(int(c**0.5), 0, -1):
        if c % a == 0:
            b = c // a
            return b, a
    return 1, c


def numpy_reshape(m: np.ndarray, dim: int = 2):
    """
    Reshapes matrix m into a desired dim-dimensional array
    :param m:
    :param dim:
    :return:
    """
    shape = m.shape

    for i in range(m.ndim, dim, -1):
        new_shape = list(shape[:-1])
        new_shape[-1] = shape[-1] * shape[-2]
        shape = new_shape

    return m.reshape(shape)


class PopMonitor(object):
    def __init__(self, populations: tuple | list,
                 variables: tuple[str] | list[str] | None = None,
                 sampling_rate: float = 2.0,
                 auto_start: bool = False):

        # define variables to track
        if variables is not None:
            assert len(populations) == len(variables), "The Arrays of populations and variables must have the same length"

            self.variables = variables
        else:
            self.variables = ['r'] * len(populations)

        # init monitors
        self.monitors = []

        for i, pop in enumerate(populations):
            self.monitors.append(ann.Monitor(pop, self.variables[i], period=sampling_rate, start=auto_start))

    def start(self):
        for monitor in self.monitors:
            monitor.start()

    def stop(self):
        for monitor in self.monitors:
            monitor.pause()

    def resume(self):
        for monitor in self.monitors:
            monitor.resume()

    def get(self, delete: bool = True, reshape: bool = True):
        res = {}

        for i, monitor in enumerate(self.monitors):
            res[self.variables[i] + '_' + monitor.object.name] = monitor.get(self.variables[i],
                                                                             keep=not delete, reshape=reshape)
        return res

    def get_specific_monitor(self, pop_name: str, delete: bool = True, reshape: bool = True):
        index = [i for i, monitor in enumerate(self.monitors) if pop_name in monitor.object.name][0]

        ret = self.monitors[index].get(self.variables[index],
                                       keep=not delete,
                                       reshape=reshape)

        return ret

    def save(self, folder, delete: bool = True):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for i, monitor in enumerate(self.monitors):
            rec = monitor.get(self.variables[i], keep=not delete, reshape=True)
            np.save(folder + self.variables[i] + '_' + monitor.object.name, rec)

    def load(self, folder):
        monitor_dict = {}

        for i, monitor in enumerate(self.monitors):
            monitor_dict[self.variables[i] + '_' + monitor.object.name] = np.load(
                folder + self.variables[i] + '_' + monitor.object.name + '.npy')

        return monitor_dict

    def animate_current_monitors(self,
                                 plot_order: tuple[int, int] | None = None,
                                 plot_types: str | list | tuple = 'Bar',
                                 fig_size: tuple[float, float] | list[float, float] = (10, 10),
                                 t_init: int = 0,
                                 save_name: str = None,
                                 label_ticks: bool = True,
                                 frames_per_sec: int | None = 10,
                                 clear_monitors: bool = False):

        results = self.get(delete=clear_monitors, reshape=True)
        PopMonitor.animate_rates(results=results,
                                 plot_types=plot_types,
                                 plot_order=plot_order,
                                 fig_size=fig_size,
                                 t_init=t_init,
                                 save_name=save_name,
                                 label_ticks=label_ticks,
                                 frames_per_sec=frames_per_sec)

    @staticmethod
    def _reshape(m: np.ndarray, dim: int = 2):
        """
        Reshapes matrix m into a desired dim-dimensional array
        :param m:
        :param dim:
        :return:
        """
        shape = m.shape

        for i in range(m.ndim, dim, -1):
            new_shape = list(shape[:-1])
            new_shape[-1] = shape[-1] * shape[-2]
            shape = new_shape

        return m.reshape(shape)

    @staticmethod
    def animate_rates(results: dict,
                      plot_order: tuple[int, int] | None = None,
                      plot_types: str | list | tuple = 'Bar',
                      fig_size: tuple[float, float] | list[float, float] = (10, 10),
                      t_init: int = 0,
                      save_name: str = None,
                      label_ticks: bool = True,
                      frames_per_sec: int | None = 10):

        """
        Animate the results of the given dictionary using various plot types.

        :param results: A dictionary containing the results to be visualized.
        :param plot_order: The order of subplots in the figure. If None, the layout is automatically determined.
        :param plot_types: The type of plot to be used for each result. If a single string is provided, it is applied to all results.
        :param fig_size: The size of the figure.
        :param t_init: The initial time step to display.
        :param save_name: The name of the file to save the animation. If None, the animation is displayed.
        :param label_ticks: Whether to label the ticks on the plots.
        :param frames_per_sec: The frames per second for the animation.
        :return:
        """
        # TODO: Making a plot type class to trim the code

        from matplotlib.widgets import Slider
        import matplotlib.animation as animation

        # define plot layout
        if plot_order is None:
            ncols, nrows = find_largest_factors(len(results))
        else:
            ncols, nrows = plot_order

        # define plot types if not defined
        if isinstance(plot_types, str):
            plot_types = [plot_types] * len(results)

        # fill the figure
        fig = plt.figure(figsize=fig_size)
        subfigs = fig.subfigures(nrows, ncols)

        plot_lists = []
        for outer_i, (subfig, key) in enumerate(zip(subfigs.flat, results)):

            # assignment plot type + key
            plot_type = plot_types[outer_i]

            # set title
            subfig.suptitle(key)
            subfig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.1, hspace=0.1)

            # good ol' switcharoo
            if plot_type == 'Matrix':
                if results[key].ndim > 4:
                    results[key] = PopMonitor._reshape(results[key])
                elif results[key].ndim == 2:
                    last_dim = results[key].shape[1]
                    inner_rows, inner_cols = find_largest_factors(last_dim)
                    results[key] = results[key].reshape(results[key].shape[0], inner_rows, inner_cols)

                res_max = np.amax(abs(results[key]))

                # subplots
                if results[key].ndim == 4:
                    last_dim = results[key].shape[-1]
                    inner_rows, inner_cols = find_largest_factors(last_dim)

                    # add subsubplots
                    axs = subfig.subplots(inner_rows, inner_cols)
                    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)

                    plots = []
                    for ax, result in zip(axs.flat, np.rollaxis(results[key][t_init], -1)):
                        p = ax.imshow(result, vmin=-res_max, vmax=res_max, cmap='RdBu',
                                      origin='lower', interpolation='none')
                        # set off tick labels for better arrangement
                        ax.set_xticks([])
                        ax.set_yticks([])

                        plots.append(p)
                else:
                    ax = subfig.subplots()
                    plots = ax.imshow(results[key][t_init], vmin=-res_max, vmax=res_max, cmap='RdBu',
                                      origin='lower', interpolation='none')

            elif plot_type == 'Plot':
                if results[key].ndim > 4:
                    results[key] = PopMonitor._reshape(results[key])
                res_max = np.amax(abs(results[key]))

                # subplots
                if results[key].ndim == 4:
                    last_dim = results[key].shape[-1]
                    inner_rows, inner_cols = find_largest_factors(last_dim)

                    # add subsubplots
                    axs = subfig.subplots(inner_rows, inner_cols)
                    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)

                    plots = []
                    for ax, result in zip(axs.flat, np.rollaxis(results[key][t_init], -1)):
                        p = ax.plot(result)
                        # set off tick labels for better arrangement
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_ylim([-ceil(res_max + 0.1, precision=1), ceil(res_max + 0.1, precision=1)])

                        plots.append(p)
                else:
                    ax = subfig.subplots()
                    plots = ax.plot(results[key][t_init])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_ylim([-ceil(res_max + 0.1, precision=1), ceil(res_max + 0.1, precision=1)])
                    ax.set_ylabel('Activity')

            elif plot_type == 'Bar':

                if results[key].ndim > 3:
                    results[key] = PopMonitor._reshape(results[key])
                res_max = np.amax(abs(results[key]))

                # subplots
                if results[key].ndim == 3:
                    last_dim = results[key].shape[-1]
                    inner_rows, inner_cols = find_largest_factors(last_dim)

                    axs = subfig.subplots(inner_rows, inner_cols)
                    plots = []
                    for ax, result in zip(axs.flat, np.rollaxis(results[key][t_init], -1)):
                        p = ax.bar(x=np.arange(1, result.shape[1] + 1, 1), height=result, width=0.5)

                        ax.set_ylabel('Activity')
                        ax.set_ylim([-ceil(res_max + 0.1, precision=1), ceil(res_max + 0.1, precision=1)])

                        plots.append(p)
                else:
                    ax = subfig.subplots()
                    plots = ax.bar(x=np.arange(1, results[key].shape[1] + 1, 1), height=results[key][t_init], width=0.5)

                    ax.set_ylabel('Activity')
                    ax.set_ylim([-ceil(res_max + 0.1, precision=1), ceil(res_max + 0.1, precision=1)])

            elif plot_type == 'Polar':

                res_max = np.amax(np.sqrt(results[key][:, 1] ** 2 + results[key][:, 2] ** 2))
                ax = subfig.add_subplot(projection='polar')

                if results[key].ndim > 2:
                    results[key] = PopMonitor._reshape(results[key])

                rad = (0, np.radians(results[key][t_init, 0]))
                r = (0, np.sqrt(results[key][t_init, 1] ** 2 + results[key][t_init, 2] ** 2))
                plots = ax.plot(rad, r)
                ax.set_ylim([0, ceil(res_max + 0.1, precision=1)])
                ax.set_ylabel([])

            elif plot_type == 'Line':

                ax = subfig.subplots()

                if results[key].ndim > 2:
                    results[key] = PopMonitor._reshape(results[key], dim=3)

                res_max = np.amax(np.abs(results[key]))

                # plotting
                ax.plot(results[key])
                plots = ax.plot(results[key][t_init], marker='x', color='r')
                ax.set_ylabel('Activity')
                ax.set_xlabel('t', loc='right')
                ax.set_ylim([-ceil(res_max + 0.1, precision=1), ceil(res_max + 0.1, precision=1)])

            elif plot_type is None:
                plots = None

            else:
                raise AssertionError('You must clarify which type of plot do you want!')

            if not label_ticks:
                plt.xticks([])
                plt.yticks([])

            plot_lists.append((key, plots, plot_type))

        # time length
        val_max = results[key].shape[0] - 1

        if save_name is None:

            ax_slider = plt.axes((0.2, 0.05, 0.5, 0.03))
            time_slider = Slider(
                ax=ax_slider,
                label='n iteration',
                valmin=0,
                valmax=val_max,
                valinit=t_init
            )

            def update(val):
                t = int(time_slider.val)
                time_slider.valtext.set_text(t)

                for key, subfigure, plt_type in plot_lists:

                    if plt_type == 'Matrix':
                        if isinstance(subfigure, list):
                            for plot, result in zip(subfigure, np.rollaxis(results[key], axis=-1)):
                                plot.set_data(result[t])
                        else:
                            subfigure.set_data(results[key][t])

                    elif plt_type == 'Plot':
                        if isinstance(subfigure, list):
                            for plot, result in zip(subfigure, np.rollaxis(results[key], axis=-1)):
                                plot.set_ydata(result[t])
                        else:
                            subfigure.set_ydata(results[key][t])

                    elif plt_type == 'Bar':
                        if isinstance(subfigure, list):
                            for plot, result in zip(subfigure, np.rollaxis(results[key], axis=-1)):
                                for j, bar in enumerate(plot):
                                    bar.set_height(result[t, j])
                        else:
                            for j, bar in enumerate(subfigure):
                                bar.set_height(results[key][t, j])

                    elif plt_type == 'Polar':
                        for line in subfigure:
                            line.set_xdata((0, np.radians(results[key][t, 0])))
                            line.set_ydata((0, np.sqrt(results[key][t, 1] ** 2 + results[key][t, 2] ** 2)))

                    elif plt_type == 'Line':
                        subfigure[0].set_ydata(results[key][t])
                        subfigure[0].set_xdata([t])

            time_slider.on_changed(update)

            plt.show()
        else:
            def update_animate(t):
                subplots = []
                for key, subfigure, plt_type in plot_lists:

                    if plt_type == 'Matrix':
                        if isinstance(subfigure, list):
                            for plot, result in zip(subfigure, np.rollaxis(results[key], axis=-1)):
                                plot.set_data(result[t])
                        else:
                            subfigure.set_data(results[key][t])

                    elif plt_type == 'Plot':
                        if isinstance(subfigure, list):
                            for plot, result in zip(subfigure, np.rollaxis(results[key], axis=-1)):
                                plot.set_ydata(result[t])
                        else:
                            subfigure.set_ydata(results[key][t])

                    elif plt_type == 'Bar':
                        if isinstance(subfigure, list):
                            for plot, result in zip(subfigure, np.rollaxis(results[key], axis=-1)):
                                for j, bar in enumerate(plot):
                                    bar.set_height(result[t, j])
                        else:
                            for j, bar in enumerate(subfigure):
                                bar.set_height(results[key][t, j])

                    elif plt_type == 'Polar':
                        for line in subfigure:
                            line.set_xdata((0, np.radians(results[key][t, 0])))
                            line.set_ydata((0, np.sqrt(results[key][t, 1] ** 2 + results[key][t, 2] ** 2)))

                    elif plt_type == 'Line':
                        subfigure[0].set_ydata(results[key][t])
                        subfigure[0].set_xdata([t])

            # make folder if not exists
            folder, _ = os.path.split(save_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            ani = animation.FuncAnimation(fig, update_animate, frames=np.arange(0, val_max))

            if save_name[-3:] == 'mp4':
                writer = animation.FFMpegWriter(fps=frames_per_sec)
            else:
                writer = animation.PillowWriter(fps=frames_per_sec)

            ani.save(save_name, writer=writer)
            plt.close(fig)

    @staticmethod
    def load_and_animate(folder: str,
                         pops: list[str] | tuple[str],
                         plot_types: list[str] | tuple[str] | str = None,
                         save_name: str | None = None):

        import glob
        # Assignment of the populations to the files
        file_list = []
        for pop in pops:
            file_list += [file for file in glob.glob(folder + 'r*.npy') if pop in file]

        # Load data
        results = {}
        for file in file_list:
            _, name = os.path.split(file)
            results[name[:-4]] = np.load(file)

        if plot_types is None:
            plot_types = 'Bar'

        PopMonitor.animate_rates(results=results,
                                 plot_types=plot_types,
                                 save_name=save_name)


class ConMonitor(object):
    def __init__(self, connections: list,
                 reshape_pre: list[bool] | None = None,
                 reshape_post: list[bool] | None = None):

        self.connections = connections
        self.weight_monitors = {}
        for con in connections:
            self.weight_monitors[con.name] = []

        if reshape_pre is None:
            self.reshape_pre = [False] * len(self.connections)
        else:
            self.reshape_pre = reshape_pre

        if reshape_post is None:
            self.reshape_post = [False] * len(self.connections)
        else:
            self.reshape_post = reshape_post

    def extract_weights(self):
        for i, con in enumerate(self.connections):

            if self.reshape_post[i] or self.reshape_pre[i]:

                pre_dim, post_dim = con.pre.geometry, con.post.geometry
                # reshape pre
                if self.reshape_pre[i] and isinstance(pre_dim, tuple):
                    pre_shape = list(pre_dim)
                elif not self.reshape_pre[i] and isinstance(pre_dim, tuple):
                    pre_shape = [np.prod(pre_dim)]
                else:
                    pre_shape = [pre_dim]

                # reshape post
                if self.reshape_post[i] and isinstance(post_dim, tuple):
                    post_shape = list(post_dim)
                elif not self.reshape_post[i] and isinstance(post_dim, tuple):
                    post_shape = [np.prod(post_dim)]
                else:
                    post_shape = [post_dim]

                weights = np.array([dendrite.w for dendrite in con]).reshape(post_shape + pre_shape)
            else:
                weights = np.array([dendrite.w for dendrite in con])

            self.weight_monitors[con.name].append(weights)

    def save_cons(self, folder: str):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for con in self.connections:
            np.save(folder + 'w_' + con.name, self.weight_monitors[con.name])

    def load_cons(self, folder: str):
        con_dict = {}
        for con in self.connections:
            con_dict['w_' + con.name] = np.load(folder + 'w_' + con.name + '.npy')

        return con_dict

    def reset(self):
        for con in self.connections:
            self.weight_monitors[con.name] = []

    def current_weight_diff(self,
                            t_init: int = 0, t_end: int = -1,
                            plot_order: tuple[int, int] | None = None,
                            fig_size: tuple[float, float] | list[float, float] = (10, 10),
                            save_name: str = None):

        ConMonitor.weight_difference(self.weight_monitors,
                                     t_init=t_init, t_end=t_end,
                                     plot_order=plot_order,
                                     fig_size=fig_size,
                                     save_name=save_name)

    @staticmethod
    def weight_difference(monitors: dict,
                          t_init: int = 0, t_end: int = -1,
                          plot_order: tuple[int, int] | None = None,
                          fig_size: tuple[float, float] | list[float, float] = (10, 10),
                          save_name: str = None):
        """
        Function to calculate the weight difference based on the provided monitors.
        """

        if plot_order is None:
            ncols, nrows = find_largest_factors(len(monitors))
        else:
            ncols, nrows = plot_order
            if nrows * ncols <= len(monitors):
                raise ValueError

        fig = plt.figure(constrained_layout=True, figsize=fig_size)
        subfigs = fig.subfigures(nrows, ncols)

        if len(monitors) > 1:
            s = subfigs.flat
        else:
            s = [subfigs]

        for outer_i, (subfig, key) in enumerate(zip(s, monitors)):
            subfig.suptitle(key)

            result = monitors[key][t_end] - monitors[key][t_init]
            # type of plot is dependent on dimension
            if result.ndim == 1:
                # plot
                ax = subfig.subplots()
                ax.plot(np.arange(len(result)) + 1, result)
            elif result.ndim == 2:
                # imshow
                ax = subfig.subplots()
                ax.imshow(result, cmap='RdBu', origin='lower', interpolation='none')
            elif result.ndim == 3:
                sub_rows, sub_cols = find_largest_factors(result.shape[0])
                axs = subfig.subplots(nrows=sub_rows, ncols=sub_cols)

                for inner_i, ax in enumerate(axs.flat):
                    ax.imshow(result[inner_i], cmap='RdBu', origin='lower', interpolation='none',
                              vmax=np.amax(result), vmin=np.amin(result))
                    ax.tick_params(axis="both", labelsize=4 + 24/result.shape[0])
            else:
                if result.ndim > 4:
                    result = numpy_reshape(result, dim=4)

                dim1 = result.shape[0]
                dim2 = result.shape[1]

                sub_rows, sub_cols = find_largest_factors(dim1 * dim2)
                axs = subfig.subplots(nrows=sub_rows, ncols=sub_cols)

                for inner_i, ax in enumerate(axs.flat):
                    ax.imshow(result[int(inner_i%dim1), int(inner_i%dim2)], cmap='RdBu', origin='lower', interpolation='none',
                              vmax=np.amax(result), vmin=np.amin(result))
                    ax.tick_params(axis="both", labelsize=4 + 24 / result.shape[0])

        if save_name is None:
            plt.show()
        else:
            folder, _ = os.path.split(save_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            plt.savefig(save_name)
            plt.close(fig)

    @staticmethod
    def load_and_plot_wdiff(folder: str,
                            cons: list[str] | tuple[str],
                            fig_size: tuple[float, float] | None = None,
                            save_name: str | None = None):

        import glob
        # Assignment of the populations to the files
        file_list = []
        for con in cons:
            file_list += [file for file in glob.glob(folder + '*.npy') if con in file]

        # Load data
        for file in file_list:
            results = {}

            _, name = os.path.split(file)
            results[name[:-4]] = np.load(file)

            if save_name is not None:
                save_name += '_' + name[:-4]

            if fig_size is None:
                fig_size = find_largest_factors(np.prod(results[name[:-4]].shape[0:1]))
                fig_size = np.array(fig_size) * 10

            ConMonitor.weight_difference(monitors=results,
                                         fig_size=fig_size,
                                         save_name=save_name)
