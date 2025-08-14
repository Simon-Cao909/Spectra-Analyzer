# Spectra-Analyzer
A tool for analyzing spectral data

Welcome to Spectra Analyzer, by me, Simon, and many other people. In particular, Nicholas Clark, Charmi Bhatt, and Jan Cami. This is a tool for analyzing spectra, and is great for anyone looking to streamline their work, make continuums easier, and overall make their life easier. This is a quick rundown of how it works.

You'll find the main code in spectraAnalyzer.py. You can import it into the file you're working in after downloading it. You will also need to download the data_getter file if you wish to use fits files to initialize the object.

To use the code
1. When creating an object from the class, you can declare it either with a wavelength (shape = (i,)) and flux array (shape = (i, j, k), with j being the y index and k being the x index), or with a list of fits files in the order of shortest to longest wavelength. Note that if you do both, the fits files will take priority over the wavelength and flux. If you declare it with fits files, you can also specify the header index of the data inside the fits files. The default is 1. You can choose to stitch your spectra together if you see any overlapping regions, and finally, you can declare your wavelength range in the format (λ_min, λ_max)
2. The class has accessors, mutators, and exporters, and you can use them however you like.
3. The class has other methods that will be discussed below

--

# Creating continuums
Currently, the class has two ways of creating continuums. Using a spline or using a polynomial. You can also set a continuum by taking an array the same shape as the flux and filling it in with continuum points. PRESS ESCAPE TO EXIT THE PLOT. PRESSING THE 'X' WILL NOT DO ANYTHING.
1. The spline method
   You call this method with obj.fit_spline((x_pixel, y_pixel), k = degree, s = how_well_to_fiit, verbose = some_int, export_directory = "directory_here").
   This will create the flux vs. wavelength plot at the specified pixel, and this provides you with an interactive interface to create a spline. The following are the things you can do:
   1. left click - Add an anchor point for ALL pixels
   2. right click - Remove an anchor point for ALL pixels
   3. ctrl + left click - Add an anchor point for ONLY this pixel
   4. ctrl + right click - Remove an anchor point for ONLY this pixel
   5. ctrl + e - Exports the continuum for ONLY this pixel to the stated directory with filename x{x_pixel}_y{y_pixel}_Spline.csv. The first column is the wavelength and the second is the flux
3. The polynomial method
   You call this method with obj.fit_poly((x_pixel, y_pixel), poly_deg = degree, weights = None or array with shape (k,), verbose = some_int, export_directory = "directory_here")
   This will create the flux vs. wavelength plot at the specified pixel, and this provides you with an interactive interface to create a polynomial fit. There will already be a continuum plotted.
   This was done by fitting a polynomial at the specified degree to all of the data using the weights provided. If you would like to ignore a section of the data (e.g. because there are features),
   you can set the weights for that region to zero. The following are the things you can do:
   1. left click - Increase the weight of the point clicked by the increment stated by the title for ALL pixels
   2. right cick - Decrease the weight of the point clicked by the increment stated by the title for ALL pixels
   3. alt + left click - Increase the increment of the weight
   4. alt + right click - Decrease the increment of the weight
   5. ctrl + left click - Increase the weight of the point clicked by the increment stated by the title for ONLY this pixel
   6. ctrl + right click - Decrease the weight of the point clicked by the increment stated by the title for ONLY this pixel
   7. There is a slider that you can use to change the degree of the polynomial
   8. There is another slider that you can use to change the radius of change of the weight. This was made so you do not have to click every single pixel.
   
5. These are things you can do for both of the above:
   ctrl + e - Exports the continuum for ONLY this pixel to the stated directory with filename x{x_pixel}_y{y_pixel}_Spline.csv or x{x_pixel}_y{y_pixel}_Poly.csv. The first column is the wavelength
   and the second is the flux
   1. ctrl + shift + e - Exports the continuum for ALL pixels in the above format
   2. ctrl + u - Saves the continuum for ONLY this pixel. Will override any current save
   3. ctrl + shift + u - Saves the continuum for ALL pixels
   4. Arrow keys - These allow you to navigate your region. For instance, using the up arrow key will rerun the function with (x_pixel, y_pixel + 1)

IF THE KEY BINDS TO YOUR OPERATING SYSTEM ARE DIFFERENT, PLEASE LET ME KNOW BY SENDING AN EMAIL TO simon.cao909@gmail.com. FEEL FREE TO CHANGE THE SOURCE CODE KEY BINDS YOURSELF AS WELL FOUND IN FUNCTIONS onkey() AND onclick() IN THE fit_spline() AND fit_poly() METHODS.

You can export the continuum you created using the .export_continuum() method. So you do not have to go through all of this again, you can use the exported file and use the set_continuum() method to automatically put it in. You can also use methods export_anchor_points() (exports it to a csv file with col1 = x_pixel, col2 = y_pixel, col3 = anch_p1, col4 = anch_p2, ...), set_anchor_points() (dictionary in the format {(x_pixel, y_pixel): [anch_p1, anch_p2...]}, export_weights(), and set_weights() (array the same shape as the flux) to edit your continuum later.

--

# Performing Analysis
The functions provided for performing an analysis of your data include:
1. Creating an integrated surface brightness map (obj.create_integrated_flux_map(vmin = some_float, vmax = some_float)).

   This will take the continuum you created to normalize the flux, subtract 1, and then integrate over the wavelength region. It will show you the map. You can use matplotlib's built-in formatting
   to change vmin and vmax however you like
2. Finding noise (obj.find_noise((x_pixel, y_pixel), no_feature_region = (λ_min, λ_max), poly_fit_deg = some_int, verbose = some_int).

   This will find the standard deviation of the normalized no_feature_region provided and return it to give you the noise. This can later be used to find the signal to noise ratio.

3. Looking at the plot. (obj.create_plot(some_param))

   This will create a plot of your data at the inputted parameter. This parameter can either be a tuple to create a wavelength vs. flux plot at the specified pixel, or it can be a float to create a
   pixel vs. flux heatmap at the specified wavelength.

   For the wavelength vs. flux plot, you can:
   1. Press 'c' to show the continuum
   2. Press 'n' to show the normalized plot
   3. Press 'm' to show the best-fit model

   For the pixel vs. flux plot, you can:
   1. Click on any point in the plot to show the wavelength vs. flux plot at that point
   2. Press 'ctrl+c' to toggle on continuum
   3. Press 'ctrl+n' to toggle on normalization
   4. Press 'ctrl+m' to toggle on the model

   NOTE: You need to have the associated items created to be able to show the associated plots! For instance, you must first fit the models to show the models. Using the keybinds without doing that
   will create unexpected results or result in an error

4. Fitting models (obj.set_models() and obj.fit_models())

   This allows you to fit models in .csv files to your data. You can specify a radial velocity range, the directory your files are in, and the pattern to follow for the filenames. This will upload
   all of the models, each shifted to a radial velocity, to the object, and then you can run obj.fit_models() to find the best-fit parameters to your data.

   Note that doing this with a lot of models may cause your computer to overheat. I will add a way to fit models in a less-intensive way in future updates.
