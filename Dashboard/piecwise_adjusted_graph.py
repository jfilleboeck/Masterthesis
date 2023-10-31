import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Create a class to handle drag events for the curve segment
class DraggableCurveSegment:
    def __init__(self, ax, x, y, sensitivity=1.0):
        self.ax = ax
        self.x = x
        self.y = y
        self.sensitivity = sensitivity
        self.line, = ax.plot(x, y, c='blue', label='Curve')
        self.press = None
        self.background = None
        self.connect()

    def connect(self):
        self.cid_press = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def closest_point(self, x, y):
        dist = (self.x - x)**2 + (self.y - y)**2
        return np.argmin(dist)

    def on_press(self, event):
        if event.inaxes != self.line.axes: return
        idx = self.closest_point(event.xdata, event.ydata)
        self.press = idx, event.xdata, event.ydata
        self.background = self.line.figure.canvas.copy_from_bbox(self.line.axes.bbox)

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.line.axes: return
        idx, x_press, y_press = self.press
        dy = event.ydata - y_press
        # Update the curve segment smoothly based on sensitivity
        for i in range(len(self.y)):
            weight = np.exp(-self.sensitivity * abs(i - idx))
            self.y[i] += dy * weight
        self.line.set_data(self.x, self.y)
        self.line.figure.canvas.restore_region(self.background)
        self.line.axes.draw_artist(self.line)
        self.line.figure.canvas.blit(self.line.axes.bbox)

    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw()


# Create a class to handle drag events for the curve segment using Linear Weighting
class DraggableCurveLinearWeight:
    def __init__(self, ax, x, y, radius=10):
        self.ax = ax
        self.x = x
        self.y = y
        self.radius = radius  # Number of points affected around the clicked point
        self.line, = ax.plot(x, y, c='blue', label='Curve')
        self.press = None
        self.background = None
        self.connect()

    def connect(self):
        self.cid_press = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def closest_point(self, x, y):
        dist = (self.x - x)**2 + (self.y - y)**2
        return np.argmin(dist)

    def on_press(self, event):
        if event.inaxes != self.line.axes: return
        idx = self.closest_point(event.xdata, event.ydata)
        self.press = idx, event.xdata, event.ydata
        self.background = self.line.figure.canvas.copy_from_bbox(self.line.axes.bbox)

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.line.axes: return
        idx, x_press, y_press = self.press
        dy = event.ydata - y_press
        # Update the curve segment smoothly based on radius
        for i in range(max(0, idx - self.radius), min(len(self.y), idx + self.radius + 1)):
            weight = 1 - abs(i - idx) / self.radius
            self.y[i] += dy * weight
        self.line.set_data(self.x, self.y)
        self.line.figure.canvas.restore_region(self.background)
        self.line.axes.draw_artist(self.line)
        self.line.figure.canvas.blit(self.line.axes.bbox)

    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw()


# Create a class to handle drag events for the curve segment using Fixed Range
class DraggableCurveFixedRange:
    def __init__(self, ax, x, y, range_size=10):
        self.ax = ax
        self.x = x
        self.y = y
        self.range_size = range_size  # Number of points affected around the clicked point
        self.line, = ax.plot(x, y, c='blue', label='Curve')
        self.press = None
        self.background = None
        self.connect()

    def connect(self):
        self.cid_press = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def closest_point(self, x, y):
        dist = (self.x - x)**2 + (self.y - y)**2
        return np.argmin(dist)

    def on_press(self, event):
        if event.inaxes != self.line.axes: return
        idx = self.closest_point(event.xdata, event.ydata)
        self.press = idx, event.xdata, event.ydata
        self.background = self.line.figure.canvas.copy_from_bbox(self.line.axes.bbox)

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.line.axes: return
        idx, x_press, y_press = self.press
        dy = event.ydata - y_press
        # Update the curve segment based on a fixed range
        for i in range(max(0, idx - self.range_size), min(len(self.y), idx + self.range_size + 1)):
            self.y[i] += dy
        self.line.set_data(self.x, self.y)
        self.line.figure.canvas.restore_region(self.background)
        self.line.axes.draw_artist(self.line)
        self.line.figure.canvas.blit(self.line.axes.bbox)

    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw()



# Create a class to handle drag events for the curve segment using Gaussian Weighting
class DraggableCurveGaussianWeight:
    def __init__(self, ax, x, y, sigma=5):
        self.ax = ax
        self.x = x
        self.y = y
        self.sigma = sigma  # Standard deviation for Gaussian function
        self.line, = ax.plot(x, y, c='blue', label='Curve')
        self.press = None
        self.background = None
        self.connect()

    def connect(self):
        self.cid_press = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def closest_point(self, x, y):
        dist = (self.x - x)**2 + (self.y - y)**2
        return np.argmin(dist)

    def on_press(self, event):
        if event.inaxes != self.line.axes: return
        idx = self.closest_point(event.xdata, event.ydata)
        self.press = idx, event.xdata, event.ydata
        self.background = self.line.figure.canvas.copy_from_bbox(self.line.axes.bbox)

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.line.axes: return
        idx, x_press, y_press = self.press
        dy = event.ydata - y_press
        # Update the curve segment smoothly based on Gaussian weighting
        for i in range(len(self.y)):
            weight = np.exp(-((i - idx) ** 2) / (2 * self.sigma ** 2))
            self.y[i] += dy * weight
        self.line.set_data(self.x, self.y)
        self.line.figure.canvas.restore_region(self.background)
        self.line.axes.draw_artist(self.line)
        self.line.figure.canvas.blit(self.line.axes.bbox)

    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw()




# Create a class to handle drag events for the curve segment using Normalized Gaussian Weighting
class DraggableCurveNormalizedGaussian:
    def __init__(self, ax, x, y, sigma=5):
        self.ax = ax
        self.x = x
        self.y = y
        self.sigma = sigma  # Standard deviation for Gaussian function
        self.line, = ax.plot(x, y, c='blue', label='Curve')
        self.press = None
        self.background = None
        self.connect()

    def connect(self):
        self.cid_press = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def closest_point(self, x, y):
        dist = (self.x - x)**2 + (self.y - y)**2
        return np.argmin(dist)

    def on_press(self, event):
        if event.inaxes != self.line.axes: return
        idx = self.closest_point(event.xdata, event.ydata)
        self.press = idx, event.xdata, event.ydata
        self.background = self.line.figure.canvas.copy_from_bbox(self.line.axes.bbox)

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.line.axes: return
        idx, x_press, y_press = self.press
        dy = event.ydata - y_press
        # Compute Gaussian weights and normalize them
        weights = np.exp(-((np.arange(len(self.y)) - idx) ** 2) / (2 * self.sigma ** 2))
        weights /= np.sum(weights)
        # Update the curve segment smoothly based on normalized Gaussian weighting
        self.y += dy * weights
        self.line.set_data(self.x, self.y)
        self.line.figure.canvas.restore_region(self.background)
        self.line.axes.draw_artist(self.line)
        self.line.figure.canvas.blit(self.line.axes.bbox)

    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw()

# Generate some example data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot and make the curve segment draggable using Normalized Gaussian Weighting
fig, ax = plt.subplots(figsize=(10, 6))
draggable = DraggableCurveNormalizedGaussian(ax, x, y, sigma=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interactive Plot Example (Drag the curve segment with Normalized Gaussian Weighting)')
plt.legend()
plt.show()

