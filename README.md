# Spectra-Analyzer
A tool for analyzing spectral data

Welcome to Spectra Analyzer, by me, Simon, and of course with the help of many other people, such as Nicholas Clark, Charmi Bhatt, and Jan Cami. This is a tool for analyzing spectra, and is great for anyone looking to streamline their work, make continuums easier, and overall make their life easier. This is a quick rundown of how it works.

You'll find the main code in spectraAnalyzer.py. You can import it into the file you're working in after downloading it.

To use the code
1. When creating an object from the class, you can declare it either with a wavelength (shape = (i,)) and flux array (shape = (i, j, k), with j being the y index and k being the x index), or with a list of fits files in the order of shortest to longest wavelength. Note that if you do both, the fits files will take priority over the wavelength and flux. If you declare it with fits files, you can also specify the header index of the data inside the fits files. The default is 1. You can choose to stitch your spectra together if you see any overlapping regions, and finally, you can declare your wavelength range in the format (λ_min, λ_max)
2. The class has accessors, mutators, and exporters, and you can use them however you like.
3. The class has other methods that will be discussed below

--

# Creating continuums
Currently, the class has two ways of creating continuums. Using a spline or using a polynomial. You can also set a continuum by taking an array the same shape as the flux and filling it in with continuum points.
1. The spline method
   You call this method with obj.fit_spline((x_pixel, y_pixel), k = degree, s = how_well_to_fiit, verbose = some_int, export_directory = "directory_here")
   This will create the flux vs. wavelength plot at the specified pixel, and this provides you with an interactive interface to create a spline. The following are the things you can do:
   left click - Add an anchor point for ALL pixels\n
   right click - Remove an anchor point for ALL pixels\n
   ctrl + left click - Add an anchor point for ONLY this pixel
   ctrl + right click - Remove an anchor point for ONLY this pixel
   ctrl + e - Exports the continuum for ONLY this pixel to the stated directory with filename x{x_pixel}_y{y_pixel}_Spline.csv. The first column is the wavelength and the second is the flux
2. The polynomial method
   You call this method with obj.fit_poly((x_pixel, y_pixel), poly_deg = degree, weights = None or array with shape (k,), verbose = some_int, export_directory = "directory_here")
   This will create the flux vs. wavelength plot at the specified pixel, and this provides you with an interactive interface to create a polynomial fit. There will already be a continuum plotted.
   This was done by fitting a polynomial at the specified degree to all of the data using the weights provided. If you would like to ignore a section of the data (e.g. because there are features),
   you can set the weights for that region to zero. The following are the things you can do:
   left click - Increase the weight of the point clicked by the increment stated by the title for ALL pixels
   right cick - Decrease the weight of the point clicked by the increment stated by the title for ALL pixels
   alt + left click - Increase the increment of the weight
   alt + right click - Decrease the increment of the weight
   ctrl + left click - Increase the weight of the point clicked by the increment stated by the title for ONLY this pixel
   ctrl + right click - Decrease the weight of the point clicked by the increment stated by the title for ONLY this pixel
   
3. These are things you can do for both of the above:
   ctrl + e - Exports the continuum for ONLY this pixel to the stated directory with filename x{x_pixel}_y{y_pixel}_Spline.csv or x{x_pixel}_y{y_pixel}_Poly.csv. The first column is the wavelength
   and the second is the flux
   ctrl + shift + e - Exports the continuum for ALL pixels in the above format
   ctrl + u - Saves the continuum for ONLY this pixel. Will override any current save
   ctrl + shift + u - Saves the continuum for ALL pixels
   Arrow keys - These allow you to navigate your region. For instance, using the up arrow key will rerun the function with (x_pixel, y_pixel + 1)

You can export the continuum you created using the .export_continuum() method. So you do not have to go through all of this again, you can use the exported file and use the set_continuum() method to automatically put it in.
You can also use methods export_anchor_points() (exports it to a csv file with col1 = x_pixel, col2 = y_pixel, col3 = anch_p1, col4 = anch_p2, ...), set_anchor_points() (dictionary in the format {(x_pixel, y_pixel): [anch_p1, anch_p2...]}, export_weights(), and set_weights() (array the same shape as the flux) to edit your continuum later.

--

# Performing Analysis
The functions provided for performing an analysis of your data include:
1. Creating an integrated surface brightness map (obj.create_integrated_flux_map(vmin = some_float, vmax = some_float))
   This will take the continuum you created to normalize the flux, subtract 1, and then integrate over the wavelength region. It will show you the map. You can use matplotlib's built-in formatting
   to change vmin and vmax however you like
3. Finding noise (obj.find_noise((x_pixel, y_pixel), no_feature_region = (λ_min, λ_max), poly_fit_deg = some_int, verbose = some_int)
   This will find the standard deviation of the normalized no_feature_region provided and return it to give you the noise. This can later be used to find the signal to noise ratio.
