// Variables for creating the plot
const plotData = [{
    x: xData,
    y: yData,
    mode: 'lines',
    type: 'scatter',
}];

const layout = {
    dragmode: 'select',
    autosize: true,
    responsive: true,
};
let selectedFeature = document.getElementById('feature-select').value;

// Variables for counterfactual axis description
const specificValue = 0.953; // Value from Table
const hoverTextArray = plotData[0].y.map(y => `=${(y - specificValue).toFixed(3)}<extra></extra>`);
plotData[0].hovertemplate = hoverTextArray;


// Create the plot and store x1 and x2 values of bounding boxes
Plotly.newPlot('plot', plotData, layout).then(() => {
    console.log("Plot created");
    store_x_values();
    createHistogramPlot();
});

function drawLines(x, y) {
    const update = {
        shapes: [
            {
                type: 'line',
                x0: x,
                y0: y,
                x1: x,
                y1: layout.yaxis.range[0],
                line: {
                    color: 'red',
                    width: 2,
                    dash: 'dot',
                },
            },
            {
                type: 'line',
                x0: x,
                y0: y,
                x1: layout.xaxis.range[0],
                y1: y,
                line: {
                    color: 'red',
                    width: 2,
                    dash: 'dot',
                },
            }
        ]
    };

    Plotly.relayout('plot', update);
}

var plotDiv = document.getElementById('plot');
// plotDiv.on('plotly_hover', function(data){
//     const xValue = data.points[0].x;
//     const yValue = data.points[0].y;
//     drawLines(xValue, yValue);
// });

function store_x_values() {
    document.getElementById('plot').on('plotly_selected', function(data) {
        if (!data) return;
        const x1 = data.range.x[0];
        const x2 = data.range.x[1];
        document.getElementById('x1-value').value = x1.toFixed(2);
        document.getElementById('x2-value').value = x2.toFixed(2);
    });
};

document.addEventListener('DOMContentLoaded', function() {
    const selectBox = document.getElementById('feature-select');
    // Event listener for feature selection change
    selectBox.addEventListener('change', function () {
        selectedFeature = selectBox.value;
        fetchFeatureData(selectedFeature);
    });
    predictAndGetMSE();
    //fetchFeatureData(selectBox.value);
    fetchDataAndCreateTable();
});

function fetchFeatureData(selected_feature) {
    fetch('/feature_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({selected_feature: selected_feature}),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
        } else {
            isNumericFeature = data.is_numeric;
            let plotX = data.x;
            let plotY = data.y;

            // Reset the axis layout
            layout.xaxis = {};
            layout.yaxis = {};

            if (!isNumericFeature) {
                // Set the x-axis layout for categorical data
                layout.xaxis = {
                    tickvals: plotX,
                    ticktext: data.original_values
                };
                Plotly.react('plot', [{
                    x: plotX,
                    y: plotY,
                    type: 'bar'
                }], layout);
            } else {
                // Update the plot for numeric data
                Plotly.react('plot', [{
                    x: plotX,
                    y: plotY,
                    type: 'scatter',
                    mode: 'lines'
                }], layout);
            }
            if (data.histogram && data.bin_edges) {
                createHistogramPlot(data.histogram, data.bin_edges);
            }

            store_x_values();
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function createHistogramPlot(histogram, bin_edges) {
    // Ensure that histogram and bin_edges are of equal length
    if (histogram.length !== bin_edges.length - 1) {
        console.error("Histogram and bin edges length mismatch");
        return;
    }

    // Prepare data for the plot
    var trace = {
        x: bin_edges,
        y: histogram.map(x => -Math.abs(x)), // Negate the values for downward extension
        type: 'bar',
        marker: {
            color: 'blue' // You can choose the color
        },
        hoverinfo: 'x+y', // Shows original y value on hover
    };

    var layout = {
        title: 'Downward Extending Histogram',
        xaxis: {
            title: 'Bins'
        },
        yaxis: {
            title: 'Frequency'
        },
        bargap: 0.05 // Adjust the gap between bars if needed
    };

    Plotly.newPlot('histogram-plot', [trace], layout);
}




function setConstantValue() {
    const x1 = parseFloat(document.getElementById('x1-value').value);
    const x2 = parseFloat(document.getElementById('x2-value').value);
    const newYValue = parseFloat(document.getElementById('new-y-value').value);

    // Store the current x-axis and y-axis range
    const gd = document.getElementById('plot');
    const currentXAxisRange = gd.layout.xaxis.range;
    const currentYAxisRange = gd.layout.yaxis.range;

    fetch('/setConstantValue', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({x1: x1, x2: x2, new_y: newYValue, selected_feature: selectedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {y: [data.y]}).then(() => {
            // Reapply the stored range values to maintain the zoom level
            Plotly.relayout(gd, {
                'xaxis.range': currentXAxisRange,
                'yaxis.range': currentYAxisRange
            });
        });
    });
}



function setMonotonicIncrease() {
    const x1 = parseFloat(document.getElementById('x1-value').value);
    const x2 = parseFloat(document.getElementById('x2-value').value);

    fetch('/monotonic_increase', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({x1: x1, x2: x2, selected_feature: selectedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {y: [data.y]});
    });
}

function setMonotonicDecrease() {
    const x1 = parseFloat(document.getElementById('x1-value').value);
    const x2 = parseFloat(document.getElementById('x2-value').value);

    fetch('/monotonic_decrease', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({x1: x1, x2: x2, selected_feature: selectedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {y: [data.y]});
    });
}

function performCubicSplineInterpolation() {
    const selectedFeature = document.getElementById('feature-select').value;

    fetch('/cubic_spline_interpolate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ selected_feature: selectedFeature })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error occurred: ' + data.error);
        } else {
            // If the spline interpolation was successful, add the new trace to the plot
            var newTrace = {
                x: data.x,
                y: data.y,
                type: 'scatter',
                mode: 'lines',
                name: 'Spline'
            };

            Plotly.addTraces('plot', newTrace);
        }
    })
}

function updateWeights() {
    const selectedFeature = document.getElementById('feature-select').value;
    console.log("that worked")
    fetch('/update_weights', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({selected_feature: selectedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {y: [data.y]});
    });
}


function predictAndGetMSE() {
    fetch('/predict_and_get_mse')
    .then(response => response.json())
    .then(data => {
        // Create the MSE output content
        let mseOutputContent = 'Mean Squared Error on Training Set: ' + data.mse_train.toFixed(2) +
                               '<br>Mean Squared Error on Validation Set: ' + data.mse_val.toFixed(2);

        // Set the content to the mse-output div and show it
        let mseOutputDiv = document.getElementById('mse-output');
        mseOutputDiv.innerHTML = mseOutputContent;
        mseOutputDiv.style.display = 'block';
    })
    .catch(error => {
        // If there's an error, display it in the mse-output div
        let mseOutputDiv = document.getElementById('mse-output');
        mseOutputDiv.className = 'alert alert-danger';
        mseOutputDiv.innerHTML = 'Error: ' + error;
        mseOutputDiv.style.display = 'block';
    });
}

function resetGraph() {
    fetch('/get_original_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({selected_feature: selectedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {x: [data.x], y: [data.y]});
    });
}

function undoLastChange() {
    fetch('/undo_last_change', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({selected_feature: selectedFeature})
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            Plotly.update('plot', {y: [data.y]});
        }
    });
}

function updateModel() {
    fetch('/update_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({selected_feature: selectedFeature})
    })
    .then(response => response.json())
    .then(data => {
        // Create the MSE output content
        let mseOutputContent = 'Mean Squared Error on Training Set: ' + data.mse_train.toFixed(2) +
                               '<br>Mean Squared Error on Testing Set: ' + data.mse_test.toFixed(2);

        // Set the content to the mse-output div and show it
        let mseOutputDiv = document.getElementById('mse-output');
        mseOutputDiv.innerHTML = mseOutputContent;
        mseOutputDiv.style.display = 'block';
    })
}
// Function to generate columns based on JSON keys
function generateColumns(data) {
    var columns = [];
    if(data.length > 0){
        for (var key in data[0]) {
            columns.push({title: key.charAt(0).toUpperCase() + key.slice(1), field: key});
        }
    }
    return columns;
}

// Function to create the table
function createTable(data) {
    var table = new Tabulator("#datagrid-table", {
        data: data,
        layout: "fitColumns", //fit columns to width of table
        columns: generateColumns(data),
    });
}

function fetchDataAndCreateTable() {
    fetch('/load_data_grid_instances', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({type_of_data: 'initial'})
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => createTable(data))
    .catch(error => console.error('Error:', error));
}
