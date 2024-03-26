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
let displayedFeature = document.getElementById('display-feature').value;

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

        const selectBox = document.getElementById('display-feature');
        // Event listener for feature selection change
        selectBox.addEventListener('change', function () {
            displayedFeature = selectBox.value;
            fetchFeatureData(displayedFeature);
        });
        predictAndGetMetrics();
        //fetchFeatureData(selectBox.value);
    fetchDataAndCreateTable();

    const validationDataButton = document.getElementById('validation-data-button');
    const instanceExplanationsButton = document.getElementById('instance-explanations-button');
    const shapeFunctionsButton = document.getElementById('shape-functions-button');
    const correlationMatrixButton = document.getElementById('correlation-matrix-button');

    validationDataButton.addEventListener('click', function() {
        hideAllContentSections();
        document.getElementById('datagrid-table').style.display = 'block';
    });

    instanceExplanationsButton.addEventListener('click', function() {
        hideAllContentSections();
        document.getElementById('instance-explanations-content').style.display = 'block';
    });

    shapeFunctionsButton.addEventListener('click', function() {
        hideAllContentSections();
        document.getElementById('shape-functions-content').style.display = 'block';
    });

    correlationMatrixButton.addEventListener('click', function() {
        hideAllContentSections();
        document.getElementById('correlation-matrix-content').style.display = 'block';
    });

    });
function fetchFeatureData(displayedFeature) {
    fetch('/feature_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({displayed_feature: displayedFeature}),
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
        body: JSON.stringify({x1: x1, x2: x2, new_y: newYValue, displayed_feature: displayedFeature})
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
        body: JSON.stringify({x1: x1, x2: x2, displayed_feature: displayedFeature})
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
        body: JSON.stringify({x1: x1, x2: x2, displayed_feature: displayedFeature})
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
        body: JSON.stringify({x1: x1, x2: x2, displayed_feature: displayedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {y: [data.y]});
    });
}

function setSmooth() {
    const x1 = parseFloat(document.getElementById('x1-value').value);
    const x2 = parseFloat(document.getElementById('x2-value').value);

    // Store the current x-axis and y-axis range
    const gd = document.getElementById('plot');
    const currentXAxisRange = gd.layout.xaxis.range;
    const currentYAxisRange = gd.layout.yaxis.range;


    fetch('/setSmooth', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({x1: x1, x2: x2, displayed_feature: displayedFeature})
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


function SplineInterpolation(selectedFeatures) {
    const displayed_feature = document.getElementById('display-feature').value;

    fetch('/cubic_spline_interpolate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ displayed_feature: displayed_feature, selectedFeatures: selectedFeatures })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error occurred: ' + data.error);
        }
        else {
            Plotly.update('plot', {
                y: [data.y]
            }).then(() => {
                predictAndGetMetrics();
            });
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing your request.');
    });
}



// New function to send retrain request
function sendRetrainRequest() {

    const displayed_feature = document.getElementById('display-feature').value;
    const elmScaleElement = document.getElementById('hidden_elmScale');
    const elmScale = parseFloat(elmScaleElement.value);
    const elmAlphaElement = document.getElementById('hidden_elmAlpha')
    const elmAlpha = parseFloat(elmAlphaElement.value);
    const nrSyntheticDataPointsElement = document.getElementById('hidden_nrSyntheticDataPoints')
    const nrSyntheticDataPoints = parseInt(nrSyntheticDataPointsElement.value);

    console.log(JSON.stringify({
        displayed_feature: displayed_feature,
        selectedFeatures: selectedFeatures,
        elmScale: elmScale,
        elmAlpha: elmAlpha,
        nrSyntheticDataPoints: nrSyntheticDataPoints
    }));


    fetch('/retrain_feature', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            displayed_feature: displayed_feature,
            selectedFeatures: selectedFeatures,
            elmScale: elmScale,
            elmAlpha: elmAlpha,
            nrSyntheticDataPoints: nrSyntheticDataPoints
        })
    })
    .then(response => {
        console.log('Response received');
        return response.json();
    })
    .then(data => {
        console.log('Data processed', data);
        if (data.error) {
            alert('Error occurred: ' + data.error);
        } else {

            Plotly.update('plot', {
                y: [data.y]
            });

        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert(error.message);
    }).then(() => {
        predictAndGetMetrics();
        document.getElementById('retrainFeatureModal').style.display = 'none'; // Hide the popup after the request
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
        body: JSON.stringify({displayed_feature: displayedFeature})
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
        body: JSON.stringify({displayed_feature: displayedFeature})
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


let selectedRowId = null;

function createTable(data) {
    var table = new Tabulator("#datagrid-table", {
        data: data,
        layout: "fitColumns",
        columns: generateColumns(data),
        // Define the context menu for each row
        rowContextMenu: [
            {
                label:"Order by nearest neighbor",
                action: function (e, row) {
                        e.preventDefault(); // Prevent the browser's context menu from appearing
                        const selectedRowId = row.getData().ID; // Store the row ID
                        const tableData = table.getData(); // Get all table data

                        // Send request to backend
                        fetch('/path/to/order_by_nearest', { // Replace '/path/to/order_by_nearest' with your actual backend endpoint
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                data: tableData,
                                selectedRowId: selectedRowId
                            }),
                        })
                        .then(response => response.json())
                        .then(orderedData => {
                            // Update the table with the newly ordered data
                            table.setData(orderedData);
                        })
                        .catch(error => console.error('Error:', error));
                    }
            },
            {
                label:"Add to plot 1",
                action: function(e, row){
                    e.preventDefault();
                    selectedRowId = row.getData().ID;
                    // Code to add to plot 1 here
                    console.log("Adding to plot 1, ID:", selectedRowId);
                }
            },
            {
                label:"Add to plot 2",
                action: function(e, row){
                    e.preventDefault();
                    selectedRowId = row.getData().ID;
                    // Code to add to plot 2 here
                    console.log("Adding to plot 2, ID:", selectedRowId);
                }
            },
        ],
    });
    console.log(data);
}


function fetchDataAndCreateTable() {
    console.log("Method called");
    fetch('/load_data_grid_instances', {
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


function hideAllContentSections() {
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.style.display = 'none';
    });
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


