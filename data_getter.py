import numpy as np
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
import csv
from reproject import reproject_interp

def loading_function(file_loc, header_index):
    '''
    This function loads in JWST MIRI and NIRSPEC fits data cubes, and extracts wavelength 
    data from the header and builds the corresponding wavelength array.
    
    Parameters
    ----------
    file_loc
        TYPE: string
        DESCRIPTION: where the fits file is located.
    header_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index to get wavelength data from in the header.

    Returns
    -------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [k,i,j] k is wavelength index, i and j are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [k,i,j] k is wavelength index, i and j are position index.
    '''
    
    #load in the data
    image_file = get_pkg_data_filename(file_loc)
    
    #header data
    science_header = fits.getheader(image_file, header_index)
    
    #wavelength data from header
    number_wavelengths = science_header["NAXIS3"]
    wavelength_increment = science_header["CDELT3"]
    wavelength_start = science_header["CRVAL3"]
    
    #constructing the ending point using given data
    #subtracting 1 so wavelength array is the right size.
    wavelength_end = wavelength_start + (number_wavelengths - 1)*wavelength_increment

    #making wavelength array, in micrometers
    wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)
    
    #extracting image data
    image_data = fits.getdata(image_file, ext=1)
    error_data = fits.getdata(image_file, ext=2)
    
    #sometimes wavelength array is 1 element short, this will fix that
    if len(wavelengths) != len(image_data):
        wavelength_end = wavelength_start + number_wavelengths*wavelength_increment
        wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)

    return wavelengths, image_data, error_data

def spectra_stitcher(wave_a, wave_b, data_a, data_b, offset=None):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to 
    the same location in the sky. It then finds which indices in the lower wavelength 
    data overlap with the beginning of the higher wavelength data, and combines 
    the 2 data arrays in the middle of this region. For this function, no scaling is done.
    
    It needs to work with arrays that may have different intervals, so it is split into 2
    to take a longer running but more careful approach if needed.
    
    Note that in the latter case, the joining is not perfect, and currently a gap
    of ~0.005 microns is present; this is much better than the previous gap of ~0.05 microns,
    as 0.005 microns corresponds to 2-3 indices.
    
    Parameters
    ----------
    wave_a
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
    wave_b
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_a
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_a.
            for [k,i,j] k is wavelength index, i and j are position index.
    data_b
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_b.
            for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_a and data_b joined together as described above.
            for [k,i,j] k is wavelength index, i and j are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    offset
        TYPE: float
        DESCRIPTION: offset applied to data
        
    '''
    
    #check if wavelength interval is the same or different
    check_a = np.round(wave_a[-1] - wave_a[-2], 4)
    check_b = np.round(wave_b[1] - wave_b[0], 4)

    if check_a == check_b:
    
        #check where the overlap is
        overlap = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]

        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_a) -1 - overlap

        #making a temp array to scale
        temp = np.copy(data_b)
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length%2 == 0: #even
            lower_index = overlap + split_index
            upper_index = split_index
            #print(lower_index, upper_index)
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            upper_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
        #apply offset
        distance = 10
        if overlap_length < 20:
            distance = 5
        if offset is None:
            offset1 = data_a[lower_index] - temp[upper_index]
            offset2 = data_a[lower_index - distance] - temp[upper_index - distance]
            offset3 = data_a[lower_index + distance] - temp[upper_index + distance]
            
            offset = (offset1 + offset2 + offset3)/3
        
        #offset = data_a[lower_index, i] - temp[upper_index, i]
        temp = temp + offset
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        
    else:
        #check where the overlap is, only works for wave_a
        overlap_a = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_a[-1] - wave_a[overlap_a]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_a = len(wave_a) -1 - overlap_a
        split_index_a = overlap_length_a/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_b = int(overlap_micron/check_b)
        split_index_b = overlap_length_b/2

        #making a temp array to scale
        temp = np.copy(data_b)
        
        #check if even or odd, do different things depending on which
        if overlap_length_a%2 == 0: #even
            lower_index = overlap_a + split_index_a
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_a + split_index_a + 0.5
            
        if overlap_length_b%2 == 0: #even
            upper_index = split_index_b
        else: #odd, so split_index is a number of the form int+0.5
            upper_index = split_index_b - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        #print(lower_index, upper_index)
        
        #apply offset
        distance = 10
        if overlap_length_a < 20 or overlap_length_b < 20:
            distance = 5
        
        if offset is None:
            offset1 = data_a[lower_index] - temp[upper_index]
            offset2 = data_a[lower_index - distance] - temp[upper_index - distance]
            offset3 = data_a[lower_index + distance] - temp[upper_index + distance]
            
            offset = (offset1 + offset2 + offset3)/3



        temp = temp + offset
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        #overlap = (overlap_a, overlap_length_b)
        overlap = (lower_index, upper_index)
    
    return image_data, wavelengths, offset

def spectra_stitcher_special(wave_a, wave_b, data_a, data_b, offset=None):
    '''
    Parameters
    ----------
    wave_a
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
    wave_b
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_a
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_a.
            for [k,i,j] k is wavelength index, i and j are position index.
    data_b
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_b.
            for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_a and data_b joined together as described above.
            for [k,i,j] k is wavelength index, i and j are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    overlap
        TYPE: integer (index) OR tuple (index)
        DESCRIPTION: index of the wavelength value in wave_a that equals the first element in wave_b. In the 
        case of the two wavelength arrays having different intervals, overlap is instead a tuple of the regular
        overlap, followed by the starting index in the 2nd array.
     
    '''
    
    #check if wavelength interval is the same or different
    check_a = np.round(wave_a[1] - wave_b[0], 4)
    check_b = np.round(wave_b[1] - wave_b[0], 4)
    
    if check_a == check_b:
    
        #check where the overlap is
        overlap = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_a) -1 - overlap
        
        #making a temp array to scale
        temp = np.copy(data_b)
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length%2 == 0: #even
            lower_index = overlap + split_index
            upper_index = split_index
            #print(lower_index, upper_index)
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            upper_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        
    else:
        #check where the overlap is, only works for wave_a
        overlap_a = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_a[-1] - wave_a[overlap_a]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_a = len(wave_a) -1 - overlap_a
        split_index_a = overlap_length_a/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_b = int(overlap_micron/check_b)
        split_index_b = overlap_length_b/2
        
        #making a temp array to scale
        temp = np.copy(data_b)
        
        #check if even or odd, do different things depending on which
        if overlap_length_a%2 == 0: #even
            lower_index = overlap_a + split_index_a
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_a + split_index_a + 0.5
            
        if overlap_length_b%2 == 0: #even
            upper_index = split_index_b
        else: #odd, so split_index is a number of the form int+0.5
            upper_index = split_index_b - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        #print(lower_index, upper_index)

        #hard coded because the offset is weird around 7.58, the usual strat wont work
            
            #using a wavelength of 7.525 roughly
        if offset is not None:
            pass
        else:
            offset = data_a[1243] - temp[11]
        temp = temp + offset

        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
    
    return image_data, wavelengths, offset

def wav_spec_file(filepath, x_start, x_end, breakearly = True, cols = (0,1)):
    xci, yci = cols

    with open(filepath, 'r') as csv_f:
        reader = csv.reader(csv_f)
        datax = []
        datay = []
        for row in reader:
            row[xci] = float(row[xci])
            row[yci] = float(row[yci])
            if row[xci] > x_start and row[xci] < x_end:
                datax.append(row[xci])
                datay.append(row[yci])
            elif row[xci] >= x_end and breakearly:
                break

    return datax, datay

def reprojecter(copy_cube, cube_to_project, path_to_reprojected_cube):
    # Load the reference cube
    ref_header = fits.getheader(copy_cube, 1)
    shape_out = (ref_header['NAXIS1'], ref_header['NAXIS2'])
    ref_wcs_2d = WCS(ref_header).celestial
    # Load the cube you want to reproject
    cube_to_reproject = fits.open(cube_to_project)[0]
    proj_data = cube_to_reproject.data
    proj_wcs_2d = WCS(cube_to_reproject.header).celestial
    # Reproject the cube
    reprojected_data, footprint = reproject_interp((proj_data, proj_wcs_2d), ref_wcs_2d, shape_out = shape_out)
    # Create a new fits file with the reprojected data
    new_header = ref_wcs_2d.to_header()
    hdu = fits.PrimaryHDU(data=reprojected_data, header=new_header)
    print(hdu)
    hdu.writeto(path_to_reprojected_cube, overwrite=True)

def reprojecter_nofits(copy_cube, cube_to_project, path_to_reprojected_cube):
    '''
    This is what I use to reproject the 2-D parameter matrices onto the HST image
    '''
    # Load the reference cube
    ref_header = fits.getheader(copy_cube, 1)
    shape_out = (ref_header['NAXIS1'], ref_header['NAXIS2'])
    ref_wcs_2d = WCS(ref_header).celestial
    # Load the cube you want to reproject
    cube_to_reproject = fits.open(r"C:\USRA_Research\Code\ch3_H-alpha_HST_to_CH3plusmap.fits")[0]
    proj_data = cube_to_project
    print(proj_data.shape)
    proj_wcs_2d = WCS(cube_to_reproject.header).celestial
    # Reproject the cube
    reprojected_data, footprint = reproject_interp((proj_data, proj_wcs_2d), ref_wcs_2d, shape_out = shape_out)
    # Create a new fits file with the reprojected data
    new_header = ref_wcs_2d.to_header()
    hdu = fits.PrimaryHDU(data=reprojected_data, header=new_header)
    print(hdu)
    hdu.writeto(path_to_reprojected_cube, overwrite=True)

# reprojecter_nofits(r"C:\USRA_Research\Code\F656N2009_tweak2F110W_drc_Gaia.fits",
#                    np.load(r"C:\USRA_Research\Data\Temperature_Map.npy", allow_pickle=True),
#                    r"C:\USRA_Research\Code\Reprojected_Temperature_Map.fits")
