// Initial Plot creation
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
    createHistogramPlot(hist_data, bin_edges);
});

function createHistogramPlot(hist_data, bin_edges) {
    // Prepare data for the plot
    console.log("Creating new Histogram Plot");
    var trace = {
        x: bin_edges,
        y: hist_data.map(x => Math.abs(x)),
        type: 'bar',
        marker: {
            color: 'blue' // You can choose the color
        },
        hoverinfo: 'x+y', // Shows original y value on hover
    };

    var layout = {
        xaxis: {
            title: 'Bins'
        },
        yaxis: {
            title: 'Frequency',
            tickmode: 'auto',
            nticks: 2
        },
        bargap: 0.05,
        height: 250// Adjust the gap between bars if needed
    };

    Plotly.newPlot('histogram-plot', [trace], layout);
}


// Event listener and feature selection


document.addEventListener('DOMContentLoaded', function() {
    const selectBox = document.getElementById('feature-select');
    // Event listener for feature selection change
    selectBox.addEventListener('change', function () {
        selectedFeature = selectBox.value;
        fetchFeatureData(selectedFeature);
    });
    predictAndGetMetrics();
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
                console.log("Categorical Feature")
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
                console.log("Numerical Feature")
                Plotly.react('plot', [{
                    x: plotX,
                    y: plotY,
                    type: 'scatter',
                    mode: 'lines'
                }], layout);
            }
            console.log("Both")
            console.log(data.bin_edges)
            if (data.hist_data && data.bin_edges) {
                createHistogramPlot(data.hist_data, data.bin_edges);
            }

            store_x_values();
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}



// User options to adjust spline functions


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

function setLinear() {
    const x1 = parseFloat(document.getElementById('x1-value').value);
    const x2 = parseFloat(document.getElementById('x2-value').value);

    // Store the current x-axis and y-axis range
    const gd = document.getElementById('plot');
    const currentXAxisRange = gd.layout.xaxis.range;
    const currentYAxisRange = gd.layout.yaxis.range;

    // Options: Inplace Interpolation,  Stepwise


    fetch('/setLinear', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({x1: x1, x2: x2, selected_feature: selectedFeature})
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

function retrainFeature() {
    const selectedFeature = document.getElementById('feature-select').value;
    fetch('/retrain_feature', {
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


function predictAndGetMetrics() {
    fetch('/predict_and_get_metrics')
    .then(response => response.json())
    .then(data => {
        let outputContent;
        if (data.task === "regression") {
            outputContent = 'MSE Training: ' + data.train_score.toFixed(2) +
                            '<br>MSE Validation: ' + data.val_score.toFixed(2);
        } else if (data.task === "classification") {
            outputContent = 'F1 Score Training: ' + data.train_score.toFixed(2) +
                            '<br>F1 Score Validation: ' + data.val_score.toFixed(2);
        }
        let OutputDiv = document.getElementById('metric-output');
        OutputDiv.innerHTML = outputContent;
        OutputDiv.style.display = 'block';
    })
    .catch(error => {
        // If there's an error, display it in the output div
        let OutputDiv = document.getElementById('metric-output');
        OutputDiv.className = 'alert alert-danger';
        OutputDiv.innerHTML = 'Error: ' + error;
        OutputDiv.style.display = 'block';
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


// Helper functions:
function store_x_values() {
    document.getElementById('plot').on('plotly_selected', function(data) {
        if (!data) return;
        const x1 = data.range.x[0];
        const x2 = data.range.x[1];
        document.getElementById('x1-value').value = x1.toFixed(2);
        document.getElementById('x2-value').value = x2.toFixed(2);
    });
};

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
        layout: "fitColumns",
        columns: generateColumns(data),
        rowClick: function(e, row) {
            // Remove highlight class from all rows
            table.getRows().forEach(function(r) {
                r.getElement().classList.remove("highlighted-row");
            });
            // Add highlight class to the clicked row
            row.getElement().classList.add("highlighted-row");
        },
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

function orderByNearest() {
    fetch('/order_by_nearest', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
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

document.getElementById('order-nearest').addEventListener('click', orderByNearest);

