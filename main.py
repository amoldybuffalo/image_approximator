import cv2
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from sympy import symbols
from numpy import linspace
from sympy import lambdify
import math 
import sys

def func1(x, a, b, c):
    return a*x**2  + b*x + c

def func2(x, a, b):
    return a*x+b

def func3(x, a,b):
    return a*math.e**x + b

def func4(x,a,b):
    a*np.sin(x)+b

def func4(x,a,b):
    w = a**2 - x**2
    if w.all() > 0:
        return np.sqrt(w) + b 
    else:
        return 0

def func5(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

functions = [func1, func2, func3, func4, func5]

def mean_square_error(a, b):
    return np.square(np.subtract(a,b)).mean() 

def calculate_polynomial(function, input):
    output = []
    for a in input:
        out = function(a)
        output.append(out)
    return output

def curve_of_best_fit(points, max_degree, independent_variable="x"):
    error = []
    function_list = []
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    if independent_variable == "x":
        p0 = x_values
        p1 = y_values
    elif independent_variable == "y":
        p0 = y_values
        p1 = x_values
    else:
        print("ERROR")
    parameters = []
    for function in functions:
        try:
            parameters = curve_fit(function, p0, p1,  maxfev=5000)
            f = lambda x : function(x, *parameters[0])
            predicted_values = calculate_polynomial(f, p0)
            error.append(mean_square_error(p1, predicted_values)) 
        except:
            error.append(100000)
        print(error)
        function_list.append((function,parameters))
    if min(error) == 100000:
        return None, None
    else:
        return function_list[error.index(min(error))]


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    threshhold = 100
    height, width = img.shape
    for i in range(0,height):
        for j in range(0, width):
            if img[i,j] < threshhold:
                img[i,j] =  0
            else:
                img[i,j] = 1 if show_original_image:
        plt.imshow(img)

def find_function(points, independent_variable="x"):
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    input = x_values if independent_variable == "x" else y_values
    f, p = curve_of_best_fit(points, 10, independent_variable)
    if f == p == None:
        return
    f_to_plot = lambda x : f(x, *p[0])
    plot_polynomial(f_to_plot, input, independent_variable)


def find_section(points, detail, independent_variable="x"):
     arr = []
     if points != []:
        if len(points) < detail:
            find_function(points, independent_variable)
        else:
            for i in range(0, len(points)):
                if (i % detail == 0 and i != 0):
                    find_function(arr, independent_variable)
                    arr = []
                elif i == len(points):
                    find_function(arr, independent_variable)
                    arr = []
                    break
                else:
                    arr.append(points[i])
                #plt.plot(top_half[i][0],top_half[i][1],'ro')

def find_all_functions(contour, detail):
    points = get_points_from_contour(contour)
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    #plt.plot(x_values, y_values)
    top_half = [point for point in points  if point[1] <= (max(y_values)+min(y_values))/2]
    bottom_half = [point for point in points if point[1] > (max(y_values)+min(y_values))/2]
    left_half = [point for point in points  if point[0] <= (max(x_values)+min(x_values))/2]
    right_half = [point for point in points  if point[0] > (max(x_values)+min(x_values))/2]
    find_section(top_half, detail, "x")
    find_section(bottom_half, detail, "x")
    find_section(left_half, detail, "y")
    #find_section(right_half, detail, "y")
        
    

def get_points_from_contour(contour):
    arr = []
    for i in range(len(contour)):
       arr.append(list(contour[i][0]))
    return arr



def approximate_image(filename, show_original_image=False):
    img = read_image("dog.png")
     if show_original_image:
        plt.imshow(img)
    contours = contours(img)
    for contour in contours:
        find_all_functions(contour, 7)
   
    if not show_original_image:
        plt.gca().invert_yaxis()
    plt.show()


   
if len(sys.argv) > 1:
    approximate_image(sys.argv[1])
else:
    print("Please put an image path as an argument.")


    
