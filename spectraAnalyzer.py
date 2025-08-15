import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wdg
import csv
from scipy.interpolate import UnivariateSpline
from scipy import integrate
import os
import lmfit as lf

# Loading_function and spectra_stitcher are functions by Nicholas Clark
from Data_getter import wav_spec_file, loading_function, spectra_stitcher

def poly_func(x, **params):
    total = 0
    for i in range(len(params)):
        total += params[f'a{i}'] * x ** i
    return total

class spectraAnalyzer:
    '''
    Welcome to Spectra Analyzer, by me, Simon, and many other people. 
    In particular, Nicholas Clark, Charmi Bhatt, and Jan Cami. 
    
    This is a tool for analyzing spectra, and is great for anyone looking to streamline their work, 
    make continuums easier, and overall make their life easier. 
    '''
    
    '''
    Special Methods
    '''
    def __init__(self, wavelength = None, flux = None, fits_filepaths = [], header_index = 1, stitch=True, wavelength_range=(-np.inf, np.inf)):        
        '''
        When creating an object from the class, you can declare it either with a wavelength (shape = (i,)) and flux array (shape = (i, j, k), 
        with j being the y index and k being the x index), or with a list of fits files in the order of shortest to longest wavelength. 
        Note that if you do both, the fits files will take priority over the wavelength and flux. 
        If you declare it with fits files, you can also specify the header index of the data inside the fits files. 
        The default is 1. You can choose to stitch your spectra together if you see any overlapping regions, 
        and finally, you can declare your wavelength range in the format (λ_min, λ_max)
        '''
        
        if len(fits_filepaths) == 1:
            data = loading_function(fits_filepaths[0])
            wavelength = data[0]
            flux = data[1]
        elif len(fits_filepaths) >= 2:
            data = []
            for ind, filepath in enumerate(fits_filepaths):
                data.append(loading_function(filepath, header_index))
            
            if stitch:
                flux, wavelength = (None, None)
                for ind, arr in enumerate(data):
                    if ind == len(data)-1:
                        break

                    if ind == 0:
                        flux, wavelength, _ = spectra_stitcher(arr[0], data[ind+1][0], arr[1], data[ind+1][1])
                    elif ind >= 1:
                        flux, wavelength, _ = spectra_stitcher(wavelength, data[ind+1][0], flux, data[ind+1][1])
            elif not stitch:
                flux, wavelength = (None, None)

                for ind, arr in enumerate(data):
                    if ind == 0:
                        wavelength, flux = (arr[0], arr[1])
                    elif ind >= 1:
                        wavelength, flux = (np.concatenate((wavelength, arr[0])), np.concatenate((flux, arr[1])))
        
        if wavelength_range[0] >= wavelength_range[1]:
            raise ValueError("The wavelength range is invalid! The first element must be less than the second")
        
        wavelength_range_mask = (wavelength >= wavelength_range[0]) * (wavelength <= wavelength_range[1])
        self._full_wavelength = wavelength
        self._full_flux = flux
        self._wavelength = wavelength[wavelength_range_mask]
        self._flux = flux[wavelength_range_mask]
        self._wavelength_range = wavelength_range

        self._continuum = np.empty_like(self._flux)
        self._anchor_points = dict()
        for y_index in range(self._flux.shape[1]):
            for x_index in range(self._flux.shape[2]):
                self._anchor_points[(x_index, y_index)] = []
        self._weights = np.ones_like(self._flux)
        self._model_data = None
        self._model_files = None
        self._radial_velocity_range = None

    def __eq__(self, other):
        return (self._flux.shape == other.get_flux().shape) == (self._wavelength == other.get_wavelength()).all() and (self._flux == other._get_flux()).all()
    
    def __str__(self):
        return f"The wavelength range is {self._wavelength_range[0]}-{self._wavelength_range[1]} μm."
    ''''''

    '''
    Accessors
    '''
    def get_wavelength(self):
        return self._wavelength
    def get_flux(self):
        return self._flux
    def get_wavelength_range(self):
        return self._wavelength_range    
    def get_continuum(self):
        return self._continuum
    def get_anchor_points(self):
        return self._anchor_points
    def get_models(self):
        return self._model_data
    ''''''

    '''
    Mutuators
    '''
    def set_wavelength(self, new_wavelength):
        self._wavelength = new_wavelength
    def set_flux(self, new_flux):
        self._flux = new_flux
        self._continuum = np.empty_like(self._flux)
        self._weights = np.ones_like(self._flux)
    def set_wavelength_range(self, new_range):
        if new_range[0] <= new_range[1]:
            raise ValueError("The wavelength range is invalid! The first element must be less than the second")
        self._wavelength_range = new_range
        new_wavelength_mask = (self._full_wavelength >= new_range[0]) & (self._full_wavelength >= new_range[1])
        self._wavelength = self._full_wavelength[new_wavelength_mask]
        self._flux = self._full_flux[new_wavelength_mask]
    def set_continuum(self, new_continuum):
        self._continuum = new_continuum
    def set_pixel_continuum(self, pixels, new_continuums):
        for ind, pixel in enumerate(pixels):
            x_index, y_index = pixel
            self._continuum[:, y_index, x_index] = new_continuums[ind]
    def set_anchor_points(self, new_anchor_points):
        self._anchor_points = new_anchor_points
    def set_anchor_pts_from_file(self, filepath):
        '''
        Sets anchor points using the csv file exported from the method .export_anchor_points()

        :param self (spectraAnalyzer): The object you are working with
        :param filepath (String): The filepath to the anchor points

        :returns: None
        '''
        anchor_pts = dict()
        with open(filepath, 'r') as csv_f:
            reader = csv.reader(csv_f)
            for lt in reader:
                key = (int(lt[0]), int(lt[1]))
                vals = []
                for ind, el in enumerate(lt):
                    if ind >= 2:
                        vals.append(int(el))
                anchor_pts[key] = vals
        self._anchor_points = anchor_pts
    def set_weights(self, new_weights):
        self._weights = new_weights
    def set_models(self, directory, radial_velocity, pattern = '', verbose=1):
        '''
        Saves the models to the object by uploading them from a provided directory

        :param self (spectraAnalyzer): The object you're working with
        :param directory (String): The directory of the model files
        :param radial velocity (1-D Array): An array of radial velocities
        :param pattern (String): A pattern that the model files follow. This can be used to only take certain files from a directory
        :param verbose (int): Decides how much of the process is outputted to the terminal
        
        :returns: None
        '''
        
        self._radial_velocity_range = radial_velocity
        wavelength = self._wavelength
        num_files = len([file for file in os.listdir(directory) if file.lower().endswith(".csv") and pattern in file])
        self._model_data = np.empty((len(radial_velocity), num_files, len(wavelength)))

        file_ind = 0
        used_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if pattern not in file:
                    continue
                filepath = os.path.join(root, file)
                used_files.append(filepath)
                model_wavelength, model_flux = wav_spec_file(filepath, 0, np.inf)
                model_wavelength = np.array(model_wavelength)
                model_flux = np.array(model_flux)

                for rv_ind, rv in enumerate(radial_velocity):
                    if verbose >= 2:
                        print(f"Currently importing file: {file}  |  v_rad = {rv} m/s")
                    shifted_model_wavelength = model_wavelength * (1 + rv/C)
                    interpolated_model_flux = np.interp(wavelength, shifted_model_wavelength, model_flux)
                    self._model_data[rv_ind, file_ind, :] = interpolated_model_flux
                
                file_ind += 1
        if verbose >= 1:
            print(f"Finished Importing Data! Shape: {self._model_data.shape}")
        self._model_files = used_files
    ''''''

    '''
    Exporters
    '''
    def export_wavelength(self, filepath, allow_pickle=True):
        np.save(filepath, self._wavelength, allow_pickle=allow_pickle)
    def export_flux(self, filepath, allow_pickle=True):
        np.save(filepath, self._flux, allow_pickle=allow_pickle)
    def export_continuum(self, filepath, allow_pickle=True):
        np.save(filepath, self._continuum, allow_pickle=allow_pickle)
    def export_anchor_points(self, filepath):
        with open(filepath, 'w', newline='') as csv_f:
            writer = csv.writer(csv_f)
            for key, item in self._anchor_points.items():
                write_this = [key[0], key[1]]
                write_this.extend(item)
                writer.writerow(write_this)
    def export_weights(self, filepath, allow_pickle=True):
        np.save(filepath, self._weights, allow_pickle=allow_pickle)
    ''''''

    def create_plot(self, parameter = None, backmap = None, datamap = None, data_with_params_filepath = None):
        '''
        This will create a plot of your data at the inputted parameter. 
        This parameter can either be a tuple to create a wavelength vs. flux plot at the specified pixel, 
        or it can be a float to create a pixel vs. flux heatmap at the specified wavelength.

        For the wavelength vs. flux plot, you can:

        Press 'c' to show the continuum
        Press 'n' to show the normalized plot
        Press 'm' to show the best-fit model
        For the pixel vs. flux plot, you can:

        Click on any point in the plot to show the wavelength vs. flux plot at that point
        Press 'ctrl+c' to toggle on continuum
        Press 'ctrl+n' to toggle on normalization
        Press 'ctrl+m' to toggle on the model

        NOTE: You need to have the associated items created to be able to show the associated plots! 
        For instance, you must first fit the models to show the models. 
        Using the keybinds without doing that will create unexpected results or result in an error

        :param self (spectraAnalyzer): The object you're working with
        :param parameter (float, tuple, or None): The parameter used to create the plot
        :param backmap (2-D array or None): If it is not None, it is used as the background for the heatmap
        :param datamap (2-D array or None): If it is not None, it is used to create a datamap on the heatmap
        :param data_with_params_filepath (String): If it is not None, it will be used as the filepath where the best-fit models are located

        :returns: None
        '''
        
        fig, ax = plt.subplots()
        if parameter is None and backmap is None:
            print("Invalid Input! Either parameter or backmap must have a value")
            return
        if isinstance(parameter, tuple):
            x_index, y_index = parameter
            
            flux = self._flux.transpose(2,1,0)[x_index, y_index]
            ax.plot(self._wavelength, flux, label = "Observations", color='black')
            continuum_plot, = ax.plot([], [], label = "Continuum", color='orange')
            normalized_plot, = ax.plot([], [], label = "Normalized Observations", color = 'black')
            model_plot, = ax.plot([], [], label = "Model", color = 'red')
            ax.set_xlabel("Wavelenth (μm)")
            ax.set_ylabel("Flux")
            ax.set_xlim(np.min(self._wavelength), np.max(self._wavelength))
            ax.set_ylim(np.nanmin(flux)-250, np.nanmax(flux)+250)
            ax.set_title(f"x = {x_index}, y = {y_index}")
            ax.legend()

            ctm_counter = 0
            norm_counter = 0
            model_counter = 0
            def onkey(event):
                nonlocal ctm_counter
                nonlocal norm_counter
                nonlocal model_counter
                print("Key Event: ", repr(event.key))
                if event.key == 'c':
                    ctm_counter += 1
                    if ctm_counter % 2 == 1:
                        print("Continuum toggle ON")
                        continuum_plot.set_data(self._wavelength, self._continuum.transpose(2,1,0)[x_index, y_index, :])
                    else:
                        print("Continuum toggle OFF")
                        continuum_plot.set_data([], [])
                elif event.key == 'n':
                    norm_counter += 1
                    if norm_counter % 2 == 1:
                        print("Showing normalized plot")
                        norm_flux = flux / self._continuum.transpose(2,1,0)[x_index, y_index, :]
                        normalized_plot.set_data(self._wavelength, norm_flux)
                        ax.set_ylim(np.nanmin(norm_flux) - 0.01, np.nanmax(norm_flux) + 0.01)
                        ax.set_ylabel("Normalized Flux")
                    else:
                        print("Returning to observations")
                        normalized_plot.set_data([], [])
                        ax.set_ylim(np.nanmin(flux)-250, np.nanmax(flux)+250)
                elif event.key == 'm':
                    model_counter += 1
                    if model_counter % 2 == 1:
                        if data_with_params_filepath is None:
                            print("No filepath inputted!")
                            return
                        norm_flux = flux / self._continuum.transpose(2,1,0)[x_index, y_index, :]
                        print("Showing model")
                        with open(data_with_params_filepath, 'r') as csv_f:
                            reader = csv.reader(csv_f)
                            next(reader)
                            data = list(reader)

                            flag = False
                            for lt in data:
                                if int(lt[0]) == x_index and int(lt[1]) == y_index:
                                    filepath = lt[2]
                                    rv = float(lt[3])
                                    redchi = float(lt[5])
                                    flag = True
                            if flag == False:
                                print("Not in the region of fitting!")
                            else:
                                modelx, modely = wav_spec_file(filepath, 0, np.inf)
                                modelx = np.array(modelx) * (1 + rv/C)
                                modely = np.array(modely)
                                model_plot.set_data(modelx, modely)
                                model_plot.set_label(f"Model: {os.path.basename(filepath)}\n$v_{{rad}}$ = {rv} m/s\n$χ_ν^2$ = {redchi:.2f}")
                                ax.set_ylim(np.nanmin(norm_flux) - 0.01, np.nanmax(norm_flux) + 0.01)
                                ax.set_ylabel("Normalized Flux")
                    else:
                        print("Returning to Observations")
                        model_plot.set_data([], [])
                        ax.set_ylim(np.nanmin(flux)-250, np.nanmax(flux)+250)

                fig.canvas.draw_idle()

            fig.canvas.mpl_connect("key_press_event", onkey)

        elif isinstance(parameter, float) or isinstance(parameter, int) or backmap is not None:
            if backmap is None:
                closest_wavelength = np.argmin(np.abs(self._wavelength-parameter))
                cw_idx = np.argmax(self._wavelength == closest_wavelength)
                flux = self._flux[cw_idx]
            else:
                flux = backmap
            ax.imshow(flux, origin='lower')
            if datamap is not None:
                ax.imshow(datamap, origin='lower')
            ax.set_xlabel("X index")
            ax.set_ylabel("Y index")

            show = 'observations'
            def onclick(event):
                if event.inaxes != ax:
                    return
                click_x, click_y = round(event.xdata), round(event.ydata)
                if event.button == 1:
                    print(f"Showing (x,y) = ({click_x}, {click_y})")
                    flux = self._flux[:, click_y, click_x]
                    fig2, ax2 = plt.subplots()
                    if show in ('observations', 'continuum'):
                        ax2.plot(self._wavelength, flux, color='black', label="Observations")
                        ax2.set_ylabel("Flux")
                        if show == 'continuum':
                            ax2.plot(self._wavelength, self._continuum[:, click_y, click_x], color='orange', label="Continuum")
                        ax2.set_ylim(flux.min()-250, flux.max()+250)
                    elif show == 'normalized':
                        norm_flux = flux / self._continuum[:, click_y, click_x]
                        ax2.plot(self._wavelength, norm_flux, color='black', label="Normalized Observations")
                        ax2.axhline(1, linestyle='--', color='red')
                        ax2.set_ylabel("Normalized Flux")
                        ax2.set_ylim(norm_flux.min()-0.01, norm_flux.max()+0.01)
                    elif show == 'model':
                        norm_flux = flux / self._continuum[:, click_y, click_x]
                        with open(data_with_params_filepath, 'r') as csv_f:
                            reader = csv.reader(csv_f)
                            next(reader)
                            data = list(reader)

                            flag = False
                            for lt in data:
                                if int(lt[0]) == click_x and int(lt[1]) == click_y:
                                    filepath = lt[2]
                                    rv = float(lt[3])
                                    redchi = float(lt[5])
                                    flag = True
                            if flag == False:
                                print("Not in the region of fitting!")
                            else:
                                modelx, modely = wav_spec_file(filepath, 0, np.inf)
                                modelx = np.array(modelx) * (1 + rv/C)
                                modely = np.array(modely)
                                ax2.plot(modelx, modely, color='red', label=f"Model: {os.path.basename(filepath)}\n$v_{{rad}}$ = {rv} m/s\n$χ_ν^2$ = {redchi:.2f}")
                        ax2.plot(self._wavelength, norm_flux, color='black', label="Normalized Observations")
                        ax2.axhline(1, linestyle='--', color='black')
                        ax2.set_ylabel("Normalized Flux")
                        ax2.set_ylim(norm_flux.min()-0.01, norm_flux.max()+0.01)
                    ax2.set_xlabel("Wavelenth (μm)")
                    ax2.set_xlim(self._wavelength.min(), self._wavelength.max())
                    ax2.set_title(f"x = {click_x}, y = {click_y}")
                    ax2.legend()
                    fig2.show()
                
            def onkey(event):
                nonlocal show
                print("Key Event:", repr(event.key))
                if event.key == 'ctrl+n':
                    print("Normalized toggle ON")
                    show = 'normalized'
                elif event.key == 'ctrl+c':
                    print("Continuum toggle ON")
                    show = 'continuum'
                elif event.key == 'ctrl+o':
                    print("Continuum toggle OFF. Normalized toggle OFF")
                    show = 'observations'
                elif event.key == 'ctrl+m':
                    if data_with_params_filepath is None:
                        print("No filepath inputted!")
                    else:
                        print("Model toggle ON")
                        show = 'model'

            fig.canvas.mpl_connect("key_press_event", onkey)
            fig.canvas.mpl_connect("button_press_event", onclick)
        
        plt.show()
        plt.close()

    '''
    Creating Continuums
    '''
    def fit_spline(self, pixel, k = 3, s = 1, verbose = 1, export_directory=None):
        '''
        This will create the flux vs. wavelength plot at the specified pixel, 
        and this provides you with an interactive interface to create a spline.

        The following are the things you can do:
        left click - Add an anchor point for ALL pixels
        right click - Remove an anchor point for ALL pixels
        ctrl + left click - Add an anchor point for ONLY this pixel
        ctrl + right click - Remove an anchor point for ONLY this pixel
        ctrl + e - Exports the continuum for ONLY this pixel to the stated directory with filename x{x_pixel}_y{y_pixel}_Spline.csv. The first column is the wavelength and the second is the flux
        ctrl + shift + e - Exports the continuum for ALL pixels in the above format
        ctrl + u - Saves the continuum for ONLY this pixel. Will override any current save
        ctrl + shift + u - Saves the continuum for ALL pixels
        t - Toggles the anchor point editing
        Arrow keys - These allow you to navigate your region. For instance, using the up arrow key will rerun the function with (x_pixel, y_pixel + 1)

        :param self (spectraAnalyzer): The object you are working with
        :param pixel (tuple): A tuple in the format (x_index, y_index)
        :param k (int): The degree of the spline
        :param s (int): The smoothness of the spline
        :param verbose (int): If set to 0, it will display nothing about the process. If set to 1, it will display something, and if set to 2, it will display everything
        :param export_directory (String or None): If not None, the files exported through ctrl + e and ctrl + shift + e will be put here. If it is None, ctrl + e and ctrl + shift + e will not work

        :returns: None
        '''
        running = True
        def plot_pixel(current_pixel):
            anchor_inds = self._anchor_points[(current_pixel[0], current_pixel[1])]
            wavelength = self._wavelength
            flux = self._flux.transpose(2,1,0)[current_pixel[0], current_pixel[1]]
            
            fig, ax = plt.subplots()
            continuum_points = {'x': [wavelength[ind] for ind in anchor_inds], 'y': [flux[ind] for ind in anchor_inds]}

            data_plot, = ax.plot(wavelength, flux, color='black', label="Observations")
            continuum_plot, = ax.plot([], [], color='orange', label="Continnuum")
            points_chosen = ax.scatter(continuum_points['x'], continuum_points['y'], color='red', s=30, label="Anchor Points")
            ax.set_xlabel("Wavelenth (μm)")
            ax.set_ylabel("Flux")
            ax.set_xlim(np.min(wavelength), np.max(wavelength))
            ax.set_ylim(np.nanmin(flux) - 250, np.nanmax(flux) + 250)
            ax.set_title(f"x = {current_pixel[0]}, y = {current_pixel[1]}")
            ax.legend()

            def update_continuum():
                if len(continuum_points['x']) < 2:
                    continuum_plot.set_data([], [])
                    fig.canvas.draw_idle()
                    return

                sorted_idcs = np.argsort(continuum_points['x'])
                x = np.array(continuum_points['x'])[sorted_idcs]
                y = np.array(continuum_points['y'])[sorted_idcs]

                if len(x) <= k:
                    contm_flux = np.interp(self._wavelength, x, y)
                else:
                    spline = UnivariateSpline(x, y, k = k, s = s)
                    contm_flux = spline(self._wavelength)
            
                continuum_plot.set_data(wavelength, contm_flux)
                points_chosen.set_offsets(np.column_stack([continuum_points['x'], continuum_points['y']]))
                fig.canvas.draw_idle()
            
            update_continuum()

            toggle = [True,0]
            def onclick(event):
                if event.inaxes != ax:
                    return
                click_x, click_y = event.xdata, event.ydata

                if not toggle[0]:
                    return

                ind_change = np.argmin(np.abs(wavelength - click_x))
                if event.button == 1 and event.key and event.key == 'control':
                    if verbose >= 2:
                        print("control + left click detected!")
                    wavelength_add = wavelength[ind_change]
                    flux_add = flux[ind_change]
                    self._anchor_points[(current_pixel[0],current_pixel[1])].append(ind_change)
                    if verbose >= 1:
                        print(f"Added: λ = {wavelength_add} μm")
                    continuum_points['x'].append(wavelength_add)
                    continuum_points['y'].append(flux_add)
                elif event.button == 3 and event.key and event.key == 'control':
                    if verbose >= 2:
                        print("control + right click detected!")
                    if len(continuum_points['x']) == 0:
                        return
                    dists = np.abs(np.array(continuum_points['x']) - click_x)
                    idx = np.argmin(dists)
                    ind_change = np.argmax(np.abs((wavelength - click_x)) == np.min(dists))
                    self._anchor_points[(current_pixel[0], current_pixel[1])].remove(ind_change)

                    if verbose >= 1:
                        print(f"Removed: λ = {continuum_points['x'][idx]} μm")
                    del continuum_points['x'][idx]
                    del continuum_points['y'][idx]
                elif event.button == 1 and not event.key:
                    if verbose >= 2:
                        print("Only left click detected")
                    for key in self._anchor_points.keys():
                        if ind_change not in self._anchor_points[key]:
                            self._anchor_points[key].append(ind_change)
                    wavelength_add = wavelength[ind_change]
                    flux_add = flux[ind_change]
                    if verbose >= 1:
                        print(f"Added: λ = {wavelength_add} μm")
                    continuum_points['x'].append(wavelength_add)
                    continuum_points['y'].append(flux_add)
                elif event.button == 3 and not event.key:
                    if verbose >= 2:
                        print("Only right click detected") # Temporary
                    if len(continuum_points['x']) == 0:
                        return
                    dists = np.abs(np.array(continuum_points['x']) - click_x)
                    idx = np.argmin(dists)
                    ind_change = np.argmax(np.abs((wavelength - click_x)) == np.min(dists))
                    for key in self._anchor_points.keys():
                        if ind_change in self._anchor_points[key]:
                            self._anchor_points[key].remove(ind_change)

                    if verbose >= 1:
                        print(f"Removed: λ = {continuum_points['x'][idx]} μm")
                    del continuum_points['x'][idx]
                    del continuum_points['y'][idx]
                    
                points_chosen.set_offsets(np.column_stack([continuum_points['x'], continuum_points['y']]))
                update_continuum()

            def onkey(event):
                if verbose >= 2:
                    print("Key event:", repr(event.key))
                if event.key == 't':
                    nonlocal toggle
                    toggle[1] += 1
                    if toggle[1] % 2 == 1:
                        print("Anchor Point Toggle OFF")
                        toggle[0] = False
                    else:
                        print("Anchor Point Toggle ON")
                        toggle[0] = True

                if event.key == "ctrl+e":
                    print(f"Exporting Continuum For x = {current_pixel[0]}, y = {current_pixel[1]}")
                    if export_directory is None:
                        print("No Directory Provided! Terminating...")
                        return
                    elif len(continuum_points['x']) == 0:
                        print("Nothing to Export! Terminating...")
                        return

                    with open(os.path.join(export_directory, f"x{current_pixel[0]}_y{current_pixel[1]}_Spline.csv"), 'w', newline='') as csv_f:
                        writer = csv.writer(csv_f)
                        contm_wavelength = wavelength
                        contm_flux = continuum_plot.get_ydata()
                        for ind, wavl in enumerate(contm_wavelength):
                            writer.writerow([wavl, contm_flux[ind]])
                    print("Done!")
                    return
                elif event.key == 'ctrl+E':
                    print("Exporting Continuum For All Pixels Using Given Anchor Points!")
                    if export_directory is None:
                        print("No Directory Provided! Terminating...")
                        return
                    elif len(continuum_points['x']) == 0:
                        print("Nothing to Export! Terminating...")
                        return
                    for y_index in range(self._flux.shape[1]):
                        for x_index in range(self._flux.shape[2]):
                            pixel_flux = self._flux.transpose(2,1,0)[x_index, y_index]
                            if np.isnan(pixel_flux).all():
                                print(f"({x_index}, {y_index}) pixel invalid due to NaN values. Skipping...")
                                continue
                            pixel_anchor_points = self._anchor_points[(x_index, y_index)]
                            x = [wavelength[point] for point in pixel_anchor_points]
                            y = [pixel_flux[point] for point in pixel_anchor_points]
                            sorted_idcs = np.argsort(x)
                            x = np.array(x)[sorted_idcs]
                            y = np.array(y)[sorted_idcs]
                            
                            pixel_spline = UnivariateSpline(x, y, k=k, s=s)

                            pixel_contm_wavelength = wavelength
                            pixel_contm_flux = pixel_spline(pixel_contm_wavelength)

                            with open(os.path.join(export_directory, f"x{x_index}_y{y_index}_Spline.csv"), 'w', newline='') as csv_f:
                                writer = csv.writer(csv_f)
                                for ind, wavl in enumerate(pixel_contm_wavelength):
                                    writer.writerow([wavl, pixel_contm_flux[ind]])
                    print("Done!")
                    return
                elif event.key == 'ctrl+u':
                    print(f"Saving Continuum For x = {current_pixel[0]}, y = {current_pixel[1]}")
                    if len(continuum_points['x']) == 0:
                        print("Nothing to Save! Terminating...")
                        return
                    contm_flux = continuum_plot.get_ydata()
                    self._continuum[:, current_pixel[1], current_pixel[0]] = contm_flux

                    print("Done!")
                    return
                elif event.key == 'ctrl+U':
                    print("Saving Continuum For All Pixels Using Given Anchor Points!")
                    if len(continuum_points['x']) == 0:
                        print("Nothing to Save! Terminating...")
                        return
                    for y_index in range(self._flux.shape[1]):
                        for x_index in range(self._flux.shape[2]):
                            pixel_flux = self._flux.transpose(2,1,0)[x_index, y_index]
                            if np.isnan(pixel_flux).all():
                                self._continuum[:, y_index, x_index] = np.nan
                                continue
                            pixel_anchor_points = self._anchor_points[(x_index, y_index)]
                            x = [wavelength[point] for point in pixel_anchor_points]
                            y = [pixel_flux[point] for point in pixel_anchor_points]
                            sorted_idcs = np.argsort(x)
                            x = np.array(x)[sorted_idcs]
                            y = np.array(y)[sorted_idcs]

                            for ind, wavl in enumerate(x):
                                idx = np.argmax(wavelength == wavl)
                                y[ind] = pixel_flux[idx]
                            
                            pixel_spline = UnivariateSpline(x, y, k=k, s=s)

                            pixel_contm_wavelength = wavelength
                            pixel_contm_flux = pixel_spline(pixel_contm_wavelength)

                            self._continuum[:, y_index, x_index] = pixel_contm_flux
                    print("Done!")
                    return
                elif event.key in ('up', 'down', 'left', 'right'):
                    plt.close(fig)
                    nonlocal pixel
                    if event.key == 'right':
                        pixel = (min(pixel[0] + 1, self._flux.shape[2]-1), pixel[1])
                    elif event.key == 'left':
                        pixel = (max(pixel[0] - 1, 0), pixel[1])
                    elif event.key == 'up':
                        pixel = (pixel[0], min(pixel[1]+1, self._flux.shape[1]-1))
                    elif event.key == 'down':
                        pixel = (pixel[0], max(pixel[1]-1, 0))
                elif event.key == 'escape':
                    plt.close(fig)
                    nonlocal running
                    running = False
                elif event.key == 'n':
                    if verbose >= 1:
                        print("Generating Normalized Plot...")
                    n_fig, n_ax = plt.subplots()
                    n_ax.plot(wavelength, flux / continuum_plot.get_ydata(), color='black')
                    n_ax.axhline(1, linestyle='--', color='red')
                    n_ax.set_xlabel("Wavelength (μm)")
                    n_ax.set_ylabel("Normalized Flux")
                    n_ax.set_title(f"x = {current_pixel[0]}, y = {current_pixel[1]} Normalized Plot")
                    n_fig.show()
                    plt.close(fig)

            cid = fig.canvas.mpl_connect("button_press_event", onclick)
            fig.canvas.mpl_connect("key_press_event", onkey)
            plt.show()
            plt.close()
        
        while running:
            plot_pixel(pixel)
    def fit_poly(self, pixel, poly_deg, weights = None, verbose = 1, export_directory=None):
        '''
        This will create the flux vs. wavelength plot at the specified pixel, 
        and this provides you with an interactive interface to create a polynomial fit. 
        There will already be a continuum plotted. This was done by fitting a polynomial 
        at the specified degree to all of the data using the weights provided. 
        If you would like to ignore a section of the data (e.g. because there are features), 
        you can set the weights for that region to zero. 
        
        The following are the things you can do:
        left click - Increase the weight of the point clicked by the increment stated by the title for ALL pixels
        right cick - Decrease the weight of the point clicked by the increment stated by the title for ALL pixels
        alt + left click - Increase the increment of the weight
        alt + right click - Decrease the increment of the weight
        ctrl + left click - Increase the weight of the point clicked by the increment stated by the title for ONLY this pixel
        ctrl + right click - Decrease the weight of the point clicked by the increment stated by the title for ONLY this pixel
        There is a slider that you can use to change the degree of the polynomial
        There is another slider that you can use to change the radius of change of the weight. This was made so you do not have to click every single pixel.
        ctrl + e - Exports the continuum for ONLY this pixel in the format x{x_pixel}_y{y_pixel}_Poly.csv
        ctrl + shift + e - Exports the continuum for ALL pixels in the above format
        ctrl + u - Saves the continuum for ONLY this pixel. Will override any current save
        ctrl + shift + u - Saves the continuum for ALL pixels
        t - Toggles the weight editing
        Arrow keys - These allow you to navigate your region. For instance, using the up arrow key will rerun the function with (x_pixel, y_pixel + 1)
        
        :param self (spectraAnalyzer): The object you are working with
        :param pixel (tuple): A tuple in the format (x_index, y_index)
        :param poly_deg (int): The initial degree of the polynomial
        :param weights (1-D array or None): The weights used to fit the polynomial. If set to None, a ones_like array will be used
        :param verbose (int): If set to 0, it will display nothing about the process. If set to 1, it will display something, and if set to 2, it will display everything
        :param export_directory (String or None): If not None, the files exported through ctrl + e and ctrl + shift + e will be put here. If it is None, ctrl + e and ctrl + shift + e will not work

        :returns: None
        '''
        
        if weights is not None:
            self._weights[:] = weights[:, None, None]
        running = True
        def plot_pixel(current_pixel):
            increment=0.5
            weights = self._weights[:, pixel[1], pixel[0]]
            wavelength = self._wavelength
            flux = self._flux.transpose(2,1,0)[current_pixel[0], current_pixel[1]]
            fitter = lf.Model(poly_func, independent_vars=['x'])
            param_names = [f'a{i}' for i in range(poly_deg+1)]
            params = lf.Parameters()
            for name in param_names:
                params.add(name, value=1)
            result = fitter.fit(flux, params=params, weights=weights, x=wavelength)

            print(result.fit_report())
            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.25)
            ax.plot(wavelength, flux, color='black', label='Observations')
            continuum_plot, = ax.plot(wavelength, result.best_fit, color='orange', label='Continuum')
            ax.set_xlabel("Wavelenth (μm)")
            ax.set_ylabel("Flux")
            ax.set_xlim(np.min(wavelength), np.max(wavelength))
            ax.set_ylim(np.nanmin(flux) - 250, np.nanmax(flux) + 250)
            ax.set_title(f"x = {current_pixel[0]}, y = {current_pixel[1]}\nWeight Increment={increment}")
            ax.legend()

            def update_continuum(new_poly_deg):
                nonlocal params
                nonlocal poly_deg

                poly_deg = new_poly_deg
                param_names = [f'a{i}' for i in range(new_poly_deg+1)]
                params = lf.Parameters()
                for name in param_names:
                    params.add(name, value=1)
                result = fitter.fit(flux, params=params, weights=weights, x = wavelength)
                continuum_plot.set_data(wavelength, result.best_fit)
                ax.set_title(f"x = {current_pixel[0]}, y = {current_pixel[1]}\nWeight Increment={increment:.1f}")
                fig.canvas.draw_idle()
            
            update_continuum(poly_deg)

            ax_slider = plt.axes([0.2, 0.15, 0.65, 0.05])
            poly_deg_slider = wdg.Slider(ax=ax_slider, label="Polynomial Degree", 
                                         valmin=1, valmax=20, 
                                         valinit=poly_deg, valstep=1)
            poly_deg_slider.on_changed(update_continuum)

            weight_edit_radius = 0
            radius_slider = wdg.Slider(ax=plt.axes([0.2, 0.05, 0.65, 0.05]), label="Weight Edit Radius (μm)", 
                                       valmin=0, valmax=5,
                                       valinit=0, valstep=0.01)
            def change_weight_edit_radius(new_radius):
                nonlocal weight_edit_radius
                weight_edit_radius = new_radius
            radius_slider.on_changed(change_weight_edit_radius)

            toggle = [True, 0]
            def onclick(event):
                if verbose == 2:
                    print("Key Press Detected:", event.key)
                if event.inaxes != ax:
                    return
                click_x, click_y = event.xdata, event.ydata

                if not toggle[0]:
                    return

                ind_change = np.argmin(np.abs(wavelength - click_x))
                change_mask = (wavelength >= (wavelength[ind_change] - weight_edit_radius)) & (wavelength <= (wavelength[ind_change] + weight_edit_radius))
                ls = np.min(wavelength[change_mask])
                rs = np.max(wavelength[change_mask])
                
                nonlocal increment
                if event.button == 1 and event.key == 'alt':
                    print("Incremented the weight increment by 0.1")
                    increment += 0.1
                elif event.button == 3 and event.key == 'alt':
                    print("Decremented the weight increment by 0.1")
                    increment = max(0, increment-0.1)

                if event.button == 1 and event.key and event.key == 'control':
                    if verbose >= 2:
                        print("control + left click detected!")
                    self._weights[change_mask, pixel[1], pixel[0]] += increment
                    if verbose >= 1:
                        print(f"λ = {ls} - {rs} μm now has weight {self._weights[ind_change, pixel[1], pixel[0]]:.1f} at this pixel")
                elif event.button == 3 and event.key and event.key == 'control':
                    if verbose >= 2:
                        print("control + right click detected!")
                    self._weights[change_mask, pixel[1], pixel[0]] = max(0, self._weights[ind_change, pixel[1], pixel[0]] - increment)
                    if verbose >= 1:
                        print(f"λ = {ls} - {rs} μm now has weight {self._weights[ind_change, pixel[1], pixel[0]]:.1f} at this pixel")
                elif event.button == 1 and not event.key:
                    if verbose >= 2:
                        print("Only left click detected")
                    self._weights[change_mask, :, :] += increment
                    if verbose >= 1:
                        print(f"Adjusted λ = {ls} - {rs} μm by +{increment:.1f} for all pixels.\nIt now has weight {self._weights[ind_change, pixel[1], pixel[0]]:.1f} at this pixel")
                elif event.button == 3 and not event.key:
                    if verbose >= 2:
                        print("Only right click detected")
                    self._weights[change_mask, :, :] = np.maximum(0, self._weights[ind_change, :, :] - increment)
                    if verbose >= 1:
                        print(f"Adjusted λ = {ls} - {rs} μm by -{increment:.1f} for all pixels.\nIt now has weight {self._weights[ind_change, pixel[1], pixel[0]]:.1f} at this pixel")
                
                nonlocal poly_deg
                update_continuum(poly_deg)
            
            def onkey(event):
                if verbose >= 2:
                    print("Key event:", repr(event.key))
                if event.key == 't':
                    nonlocal toggle
                    toggle[1] += 1
                    if toggle[1] % 2 == 1:
                        print("Anchor Point Toggle OFF")
                        toggle[0] = False
                    else:
                        print("Anchor Point Toggle ON")
                        toggle[0] = True

                if event.key == "ctrl+e":
                    print(f"Exporting Continuum For x = {current_pixel[0]}, y = {current_pixel[1]}")
                    if export_directory is None:
                        print("No Directory Provided! Terminating...")
                        return

                    with open(os.path.join(export_directory, f"x{current_pixel[0]}_y{current_pixel[1]}_Poly.csv"), 'w', newline='') as csv_f:
                        writer = csv.writer(csv_f)
                        contm_wavelength = wavelength
                        contm_flux = continuum_plot.get_ydata()
                        for ind, wavl in enumerate(contm_wavelength):
                            writer.writerow([wavl, contm_flux[ind]])
                    print("Done!")
                    return
                elif event.key == 'ctrl+E':
                    print("Exporting Continuum For All Pixels!")
                    if export_directory is None:
                        print("No Directory Provided! Terminating...")
                        return
                    for y_index in range(self._flux.shape[1]):
                        for x_index in range(self._flux.shape[2]):
                            pixel_wavelength = self._wavelength
                            pixel_flux = self._flux.transpose(2,1,0)[x_index, y_index]
                            pixel_nan_mask = np.isnan(pixel_flux)
                            if pixel_nan_mask.all():
                                print(f"({x_index}, {y_index}) pixel invalid due to NaN values. Skipping...")
                                continue
                            elif pixel_nan_mask.any():
                                print(f"({x_index}, {y_index}) pixel has certain NaN values. Will interpolate over...")
                                pixel_flux[pixel_nan_mask] = np.interp(np.flatnonzero(pixel_nan_mask), np.flatnonzero(~pixel_nan_mask), pixel_flux[~pixel_nan_mask]) 

                            result = fitter.fit(pixel_flux, params=params, weights=self._weights[:, y_index, x_index], x = pixel_wavelength)

                            with open(os.path.join(export_directory, f"x{x_index}_y{y_index}_Poly.csv"), 'w', newline='') as csv_f:
                                writer = csv.writer(csv_f)
                                for ind, wavl in enumerate(pixel_wavelength):
                                    writer.writerow([wavl, result.best_fit[ind]])
                    print("Done!")
                    return
                elif event.key == 'ctrl+u':
                    print(f"Saving Continuum For x = {current_pixel[0]}, y = {current_pixel[1]}")
                    contm_flux = continuum_plot.get_ydata()
                    self._continuum[:, current_pixel[1], current_pixel[0]] = contm_flux

                    print("Done!")
                    return
                elif event.key == 'ctrl+U':
                    print("Saving Continuum For All Pixels!")
                    transposed_flux = self._flux.transpose(2,1,0)
                    for y_index in range(self._flux.shape[1]):
                        for x_index in range(self._flux.shape[2]):
                            pixel_wavelength = self._wavelength
                            pixel_flux = transposed_flux[x_index, y_index]
                            pixel_nan_mask = np.isnan(pixel_flux)
                            if pixel_nan_mask.all():
                                print(f"({x_index}, {y_index}) pixel invalid due to NaN values. Skipping...")
                                continue
                            elif pixel_nan_mask.any():
                                print(f"({x_index}, {y_index}) pixel has certain NaN values. Will interpolate over...")
                                pixel_flux[pixel_nan_mask] = np.interp(np.flatnonzero(pixel_nan_mask), np.flatnonzero(~pixel_nan_mask), pixel_flux[~pixel_nan_mask]) 

                            result = fitter.fit(pixel_flux, params=params, weights=self._weights[:, y_index, x_index], x = pixel_wavelength)

                            self._continuum[:, y_index, x_index] = result.best_fit
                    print("Done!")
                    return
                elif event.key in ('up', 'down', 'left', 'right'):
                    plt.close(fig)
                    nonlocal pixel
                    if event.key == 'right':
                        pixel = (min(pixel[0] + 1, self._flux.shape[2]-1), pixel[1])
                    elif event.key == 'left':
                        pixel = (max(pixel[0] - 1, 0), pixel[1])
                    elif event.key == 'up':
                        pixel = (pixel[0], min(pixel[1]+1, self._flux.shape[1]-1))
                    elif event.key == 'down':
                        pixel = (pixel[0], max(pixel[1]-1, 0))
                elif event.key == 'escape':
                    plt.close(fig)
                    nonlocal running
                    running = False
                elif event.key == 'n':
                    if verbose >= 1:
                        print("Generating Normalized Plot...")
                    n_fig, n_ax = plt.subplots()
                    n_ax.plot(wavelength, flux / continuum_plot.get_ydata(), color='black')
                    n_ax.axhline(1, linestyle='--', color='red')
                    n_ax.set_xlabel("Wavelength (μm)")
                    n_ax.set_ylabel("Normalized Flux")
                    n_ax.set_title(f"x = {current_pixel[0]}, y = {current_pixel[1]} Normalized Plot")
                    n_fig.show()
                    plt.close(fig)
                
            cid = fig.canvas.mpl_connect("button_press_event", onclick)
            fig.canvas.mpl_connect("key_press_event", onkey)
            plt.show()
            plt.close()
        
        while running:
            plot_pixel(pixel)
    ''''''

    '''
    Modelling the Data
    '''
    def fit_models(self, pixels, export_filepath, n_params = 3, verbose=1, CPU_usage='medium', chi_export_directory = None):
        model_data = self._model_data
        model_files = self._model_files
        rad_vel_range = self._radial_velocity_range
        wavelength = self._wavelength
        continuum = self._continuum
        flux = self._flux
        dof = len(wavelength) - n_params

        if CPU_usage == 'medium':
            with open(export_filepath, 'w', newline='') as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(["X", "Y", "File", "Radial Velocity (m/s)", "Chi Square", "Reduced Chi Square"])
                for pixel_ind, pixel in enumerate(pixels):
                    if verbose >= 2:
                        print("Currently Processing:", pixel)
                    x_index, y_index = pixel
                    norm_pixel_flux = flux[:, y_index, x_index] / continuum[:, y_index, x_index]
                    chi_squared = np.sum(((model_data - norm_pixel_flux) / np.std(norm_pixel_flux))**2, axis=-1)
                    rv_ind, file_ind = np.unravel_index(np.argmin(chi_squared), chi_squared.shape)
                    writer.writerow([x_index, y_index, model_files[file_ind], rad_vel_range[rv_ind], chi_squared[rv_ind, file_ind], chi_squared[rv_ind, file_ind] / dof])
                    
                    if chi_export_directory is not None:
                        np.save(os.path.join(chi_export_directory, f"x{x_index}_y{y_index}_Chi2.npy"), chi_squared, allow_pickle=True)

        elif CPU_usage == 'high':
            reformatted_model_data = np.repeat(model_data[:, :, :, np.newaxis], len(pixels), axis=-1).transpose(0,1,3,2)
            reformatted_norm_flux = np.empty(reformatted_model_data.shape[2:])

            for pixel_ind, pixel in enumerate(pixels):
                x_index, y_index = pixel
                reformatted_norm_flux[pixel_ind, :] = self._flux[:, y_index, x_index] / self._continuum[:, y_index, x_index]
            
            dof = len(self._wavelength) - n_params
            chi_squared = np.sum(((reformatted_model_data - reformatted_norm_flux) / np.std(reformatted_norm_flux, axis=-1, keepdims=True)) ** 2, axis=-1)

            with open(export_filepath, 'w', newline='') as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(["X", "Y", "File", "Radial Velocity (m/s)", "Chi Square", "Reduced Chi Square"])
                for pixel_ind, pixel in enumerate(pixels):
                    if verbose >= 2:
                        print("Currently Processing", pixel)
                    rv_ind, file_ind = np.unravel_index(np.argmin(chi_squared[:, :, pixel_ind]), chi_squared.shape[:2])
                    writer.writerow([pixel[0], pixel[1], model_files[file_ind], rad_vel_range[rv_ind], chi_squared[rv_ind, file_ind, pixel_ind], chi_squared[rv_ind, file_ind, pixel_ind] / dof])
                    if False and pixel == (60,69): # Debugging
                        print(chi_squared[rv_ind, file_ind, pixel_ind] / dof)
                        rv_ind2 = [ind for ind, rv in enumerate(rad_vel_range) if rv == -43000][0]
                        file_ind2 = [ind for ind, file in enumerate(model_files) if "CO2_626_CDSD_v1.000_NTH_gauss_T30.000_N16.800_R3500.00_O2_etau.csv" in file][0]
                        print(-43000 in rad_vel_range)
                        print("A!", chi_squared[rv_ind2, file_ind2, pixel_ind] / dof)

                        mod_x, mod_y = wav_spec_file(self._model_files[file_ind],0,np.inf)
                        mod_x = np.array(mod_x) * (1 + rad_vel_range[rv_ind]/C)
                        mod_y = np.array(mod_y)

                        mod_x2, mod_y2 = wav_spec_file(self._model_files[file_ind2],0,np.inf)
                        mod_x2 = np.array(mod_x2) * (1 + rad_vel_range[rv_ind2]/C)
                        mod_y2 = np.array(mod_y2)

                        norm_flux = self._flux.transpose(2,1,0)[60,69] / self._continuum.transpose(2,1,0)[60,69]
                        plt.plot(self._wavelength, norm_flux + 0.5)
                        plt.plot(self._wavelength, norm_flux)
                        plt.plot(mod_x, mod_y + 0.5, label = "Fitted model")
                        plt.plot(mod_x2, mod_y2, label = "Other model")
                        norm_flux = self._flux[:, pixel[1], pixel[0]] / self._continuum[:, pixel[1], pixel[0]]
                        print(np.sum(((norm_flux - np.interp(self._wavelength, mod_x2, np.array(mod_y2))) / np.std(norm_flux)) ** 2) / dof)
                        plt.legend()
                        plt.show()
                        plt.close()
            if chi_export_directory is not None:
                np.save(os.path.join(chi_export_directory, "All_Pixels_Chi2.npy"), chi_squared, allow_pickle=True)
    ''''''

    def create_integrated_flux_map(self, vmin, vmax):
        '''
        Creates an integrated flux map by using the continuum to normalize the flux, subtracting one, 
        and then integrating over the wavelength region

        :param self (spectraAnalyzer): The object you're working with
        :param vmin (float): The minimum value of the heatcolor shown
        :param vmax (float): The maximum value of the heatcolor shown

        :returns: None
        '''
        
        rti_flux = self._flux / self._continuum - 1
        integral_values = integrate.trapezoid(rti_flux, x=self._wavelength, dx=0.0001, axis=0)
        fig, ax = plt.subplots()

        im = ax.imshow(integral_values, origin='lower', vmin=vmin, vmax=vmax)
        fig.colorbar(im, label="Integral Value")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
        plt.show()
    
    def find_noise(self, pixel, no_feature_region, poly_fit_deg = 1, verbose=0):
        '''
        Takes a small specified region with no features from the spectrum, normalizes it,
        and returns the standard deviation

        :param self (spectraAnalyzer): The object you're working with
        :param pixel (tuple): A tuple containing the pixel to analyze (x_index, y_index)
        :param no_feature_region (tuple): A tuple containing (lambda_min, lambda_max) of a small region with no visible features
        :param poly_fit_deg (int): The polynomial degree that is fitted to the small region
        :param verbose (int): Decides how much of the process is outputted

        :returns: The standard deviation of the normalized region specified as a NumPy float64
        '''
        
        wavelength = self._wavelength
        flux = self._flux.transpose(2,1,0)[pixel[0], pixel[1]]

        if (not isinstance(no_feature_region, tuple)) or no_feature_region[0] >= no_feature_region[1]:
            print("Invalid input for no_feature_region. Must be a tuple of (wavelength_min, wavelength_max)")
            return
        
        no_feature_mask = (wavelength >= no_feature_region[0]) & (wavelength <= no_feature_region[1])
        no_feature_wavelength = wavelength[no_feature_mask]
        no_feature_flux = flux[no_feature_mask]

        fitter = lf.Model(poly_func, independent_vars=['x'])
        param_names = [f'a{i}' for i in range(poly_fit_deg+1)]
        params = lf.Parameters()
        for name in param_names:
            params.add(name, value=1)

        result = fitter.fit(no_feature_flux, params=params, x = no_feature_wavelength)

        if verbose == 1:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.plot(wavelength, flux, "Full Observations")
            ax1.plot(no_feature_wavelength, no_feature_flux, label = "Region Used")
            ax1.legend()

            ax2.plot(no_feature_wavelength, no_feature_flux, label = "Region Used")
            ax2.plot(no_feature_wavelength, result.best_fit, label = "Best Fit")
            ax2.legend()

            plt.show()
            plt.close()

        return np.std(no_feature_flux / result.best_fit, dtype = np.float64)

    def user_friendly_run(self):
        print("Welcome to Spectra Analyzer! What would you like to do?")
        user_action = input("OPTIONS: Export, Create Plot, Fit Spline, Fit Polynomial, Fit Models\nCreate Integrated Flux Map, EXIT\n")
        
        while user_action.lower() != 'exit':
            if user_action.lower() == 'export':
                user_action2 = input("What would you like to export? (OPTIONS: Wavelength (1), Flux (2), Continuum (3), Anchor Points (4), Weights (5), EXIT)\n")
                while user_action2.lower() != 'exit':
                    filepath = input("What filepath would you like the file to have?\n")
                    exporters = [self.export_wavelength, self.export_flux, self.export_continuum, self.export_anchor_points, self.export_weights]
                    
                    if not user_action2.isnumeric():
                        print("Please input a number!")
                        continue
                    else:
                        if int(user_action2) <= 0 or int(user_action2) >= len(exporters):
                            print("Out of bounds!")
                            continue

                    exporters[int(user_action2)-1](filepath)
                    user_action2 = input("What else would you like to export? (OPTIONS: Wavelength (1), Flux (2), Continuum (3), Anchor Points (4), Weights (5), EXIT)\n")
            elif user_action.lower() == 'create plot':
                user_action2 = input("Flux vs. Wavelength (1), Heatmap (2)\n")
                if not user_action2.isnumeric():
                    print("Please input a number!")
                elif user_action2 == '1':
                    x_index = int(input("What x index? "))
                    y_index = int(input("What y index? "))
                    filepath = input("Data with params filepath? (If you do not have it, input 'no')\n")
                    if filepath.lower() == 'no':
                        filepath = None
                    else:
                        filepath = filepath.strip('"')
                    self.create_plot((x_index, y_index), data_with_params_filepath=filepath)
                elif user_action2 == '2':
                    wavelength = float(input("What wavelength? "))
                    filepath = input("Data with params filepath? (If you do not have it, input 'no')\n")
                    if filepath.lower() == 'no':
                        filepath = None
                    else:
                        filepath = filepath.strip('"')
                    self.create_plot(wavelength, data_with_params_filepath=filepath)
            elif user_action.lower() == 'fit spline':
                x_index = int(input("What x index? "))
                y_index = int(input("What y index? "))
                poly_deg = int(input("What spline degree? "))
                smoothness = int(input("How smooth? (INT) "))
                verbose = int(input("Verbose? (INT) "))
                export_dir = input("Export directory? (If None, just press enter) ")
                if export_dir == '':
                    export_dir = None
                self.fit_spline((x_index, y_index), k = poly_deg, s = smoothness, verbose=verbose, export_directory=export_dir)
            elif user_action.lower() == 'fit polynomial':
                x_index = int(input("What x index? "))
                y_index = int(input("What y index? "))
                poly_deg = int(input("What initial degree? "))
                verbose = int(input("Verbose? (INT) "))
                export_dir = input("Export directory? (If None, just press enter) ")
                if export_dir == '':
                    export_dir = None
                self.fit_poly((x_index, y_index), poly_deg, weights=None, verbose=verbose, export_directory=export_dir)
            elif user_action.lower() == 'fit models':
                directory = input("Model directory? ")
                min_radvel = int(input("Most negative radial velocity (m/s): "))
                max_radvel = int(input("Most positive radial velocity (m/s): "))
                step = int(input("What radial velocity step (m/s)? "))
                pattern = input("Is there any substring that is in every model filename? (If None, just press enter) ")

                min_x = int(input("Mininum x: "))
                max_x = int(input("Maximum x: "))
                min_y = int(input("Minimum y: "))
                max_y = int(input("Maximum y: "))

                pixels = []
                for x_index in range(min_x, max_x+1):
                    for y_index in range(min_y, max_y+1):
                        pixels.append((x_index, y_index))

                n_params = int(input("How many parameters are being fit? (INT) "))

                export_filepath = input("Export data_with_params filepath: ")
                chi_cube_dir = input("Chi cube export directory: (if None, just press enter) ")
                if chi_cube_dir == '':
                    chi_cube_dir = None

                export_filepath = export_filepath.strip('"')
                rad_vel_range = np.arange(min_radvel, max_radvel, step)


                self.set_models(directory, rad_vel_range, pattern=pattern, verbose=1)
                self.fit_models(pixels=pixels, export_filepath=export_filepath, n_params=n_params, verbose=1, CPU_usage='medium', chi_export_directory=chi_cube_dir)
            
            elif user_action.lower() == 'create integrated flux map':
                self.create_integrated_flux_map(-0.004, 0.004)
        
            user_action = input("What else would you like to do? (OPTIONS: Export, Create Plot, Fit Spline, Fit Polynomial, Fit Models\nCreate Integrated Flux Map, EXIT)\n")

'''
Example of using it
'''
# mySpec = spectraAnalyzer(fits_filepaths=[r"C:\USRA_Research\Code\ngc6302_ch1-short_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch1-medium_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch1-long_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch2-short_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch2-medium_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch2-long_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch3-short_s3d.fits", 
#                                         r"C:\USRA_Research\Code\ngc6302_ch3-medium_s3d.fits",
#                                         r"C:\USRA_Research\Code\ngc6302_ch3-long_s3d.fits"], stitch=True, wavelength_range=(9.63,10.25))
# mySpec.fit_poly((60,69), 4, None)
# mySpec.create_integrated_flux_map(vmin=-0.004, vmax=0.0005) # Integrated surface brightness map


# mySpec = spectraAnalyzer(fits_filepaths=[r"C:\USRA_Research\Code\ngc6302_ch1-short_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch1-medium_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch1-long_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch2-short_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch2-medium_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch2-long_s3d.fits",
#                                          r"C:\USRA_Research\Code\ngc6302_ch3-short_s3d.fits", 
#                                         r"C:\USRA_Research\Code\ngc6302_ch3-medium_s3d.fits",
#                                         r"C:\USRA_Research\Code\ngc6302_ch3-long_s3d.fits"], stitch=True, wavelength_range=(14.76,15.2))
# mySpec.fit_spline((60,69), export_directory=r"C:\USRA_Research\Temporary") # Creates a spline
# mySpec.create_integrated_flux_map(vmin=-0.004, vmax=0.0005) # Integrated surface brightness map














