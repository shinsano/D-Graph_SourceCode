var appBusy = true;

// alert
var alertEnabled = false;
function showAlert(message, messageType = "success") {
    if (alertEnabled) return;
    alertEnabled = true;
    ++appBusy;
    $('#alert').removeClass('alert-success');
    $('#alert').removeClass('alert-danger');
    $('#alert').addClass('alert-' + messageType);
    $('#alert-text').text(message);
    $('#alert').css('visibility', 'visible');
    $('#alert').alert();
    setTimeout(function(){$('#alert').alert('dispose');
    $('#alert').css('visibility', 'hidden');
        alertEnabled = false;
    }, 5000);
    --appBusy;
}

// internal states
var graphDataNextGroup = 0;
var inPractice = false;
var withDictionary = false;
var designBriefPractice = "You are designing a minivan for a mom with three kids.  She wants to overwrite an uncool “soccer mom” image and show off her clever choice.";
var designBriefA = "Imagine a car for the megalopolis of tomorrow and consider four important aspects: environmental friendliness, social harmony, interactive mobility, and economic efficiency.";
var designBriefB = "Design a vehicle to improve mobility for low-income individuals with physical disabilities. Help the user move independently across difficult, uneven, narrow or inclined terrain.";
var designBrief = designBriefPractice;
var inTraining = true;
var urlHashString = window.location.hash.substr(1);
var numSetWords = 0;

if (urlHashString == "DA") {
    designBrief = designBriefA;
    withDictionary = true;
} else if (urlHashString == "DB") {
    designBrief = designBriefB;
    withDictionary = true;
} else if (urlHashString == "DP") {
    designBrief = designBriefPractice;
    withDictionary = true;
    inPractice = true;
} else if (urlHashString == "GA") {
    designBrief = designBriefA;
    withDictionary = false;
} else if (urlHashString == "GB") {
    designBrief = designBriefB;
    withDictionary = false;
} else if (urlHashString == "GP") {
    designBrief = designBriefPractice;
    withDictionary = false;
    inPractice = true;
}
if (inPractice) {
    $("#stopwatch-button").text("Start Practice");
}

// SE
function sound(src) {
    this.sound = document.createElement("audio");
    this.sound.src = src;
    this.sound.setAttribute("preload", "auto");
    this.sound.setAttribute("controls", "none");
    this.sound.style.display = "none";
    document.body.appendChild(this.sound);
    this.play = function () {
        ++appBusy;
        this.sound.cloneNode().play();
        --appBusy;
    };
    this.stop = function () {
        this.sound.pause();
    };
}

$.ajaxSetup({ async: true, timeout: 10000 });

// parameters
const wordRegexPattern = /^[a-z\-]+$/;
const controlPanelWidth = 576; //$("#control-panel").width();
const wrapperRightPadding = 0;
const navbarHeight = $("#navbar").innerHeight();
const chargeStrength = -50;
var soundEffect_started = new sound("../sound/golf0.mp3");
var soundEffect_wordSet = new sound("../sound/golf1.mp3");
const draggableWordWidth = 128;
const draggableWordHeight = 24;
const maxScaledDistance = 100.0;
const minScaledDistance = 0.0;

// global variables
var word1 = null; // The following four variables corresponds to the words in the Character Space a.k.a. Design Concept Map.
var word2 = null;
var word3 = null;
var word4 = null;
var debugMode = false;
var selectedNode = null;
var selectedWord = null;
var dragInitialX = 0;
var dragInitialY = 0;
var pageX = 0;
var pageY = 0;
var dragMoved = false;
var draggedWord = null;
var draggedWordElement = null;
var relatedWordsCache = {};
var isCacheBeingLoaded = false;
var newWords = [];
var timer_startTime = 0;
var timer_elapsedTime = 0;
var timer_interval;

// D&D
const isDescendant = (e, parentID) => {
    var isChild = false;
    if (e.id === parentID) isChild = true;
    while (e == e.parentNode)
        if (e.id == parentID)
            isChild = true;
    return isChild;
};

// Graph
var numDimensions = 3; // 3 or 2
var graphData = {
    nodes: [],
    links: []
};
var graph = null; // An instance of <SVG />

function clearGraph() {
    if (graph) {
        graphData = {
            nodes: [],
            links: []
        };
        graph.graphData(graphData);
    }
}

function toggleDimentionSwitch() {
    $("#graph-3d-button").toggleClass("btn-outline-secondary");
    $("#graph-3d-button").toggleClass("btn-secondary");
    $("#graph-2d-button").toggleClass("btn-outline-secondary");
    $("#graph-2d-button").toggleClass("btn-secondary");
}

function toggleDimention() {
    graph.pauseAnimation();
    graph.cooldownTime(0);
    numDimensions = (numDimensions == 2) ? 3 : 2;
    graph.graphData({
        nodes: graphData.nodes.map(function (node) { return { id: node.id, group: node.group }; }),
        links: graphData.links.map(function (link) { return { source: link.source, target: link.target, distance: link.distance }; })
    });
    initializeGraph();
    toggleDimentionSwitch();
}

var width = window.innerWidth - controlPanelWidth - wrapperRightPadding;
var height = window.innerHeight - navbarHeight;

function setComponentSizes() {
    if (appBusy)
        return;
    ++appBusy;
    $("#body").width(window.innerWidth);
    $("#body").height(window.innerHeight);

    $("#wrapper").width(window.innerWidth - wrapperRightPadding);
    $("#wrapper").height(window.innerHeight);
    var navbarHeight = $("#navbar").innerHeight();

    width = window.innerWidth - controlPanelWidth - wrapperRightPadding;
    height = window.innerHeight - navbarHeight;
    $("#graph").css("left", 0);
    $("#graph").css("top", navbarHeight);
    $("#graph").width(width);
    $("#graph").height(height);
    if (graph) {
        graph
            .width(width)
            .height(height);
    }

    $("iframe").css("left", 0);
    $("iframe").css("top", navbarHeight);
    $("iframe").width(width);
    $("iframe").height(height);

    $("#control-panel").css("left", width);
    $("#control-panel").css("top", navbarHeight);
    $("#control-panel").width(controlPanelWidth);
    $("#control-panel").height(height);

    $("#alert").css("left", 0);
    $("#alert").css("top", navbarHeight);
    $("#alert").outerWidth(width);
    $("#alert").outerHeight($("#character-space-heading").outerHeight());
    --appBusy;
}

function calculateDistanceFromCamera(c, x, y, z) {
    var dist = (c.x - x) * (c.x - x);
    dist += (c.y - y) * (c.y - y);
    dist += (c.z - z) * (c.z - z);
    dist = Math.sqrt(dist);
    dist = Math.max(dist, 300) - 300;
    dist = dist / 500;
    dist = 1 - dist;
    return dist;
}

// Create a DIV element and add it to $("#dragged-word-container").
function createDraggedWord(e, w) {
    ++appBusy;
    draggedWord = w;
    draggedWordElement = $("<div />", {
        id: "dragged-word",
        class: "draggable-word",
        text: draggedWord,
        css: {
            "color": "black",
            "background-color": "white",
            "position": "fixed",
            "left": "" - Math.floor((e.pageX - draggableWordWidth) / 2) + "px",
            "top": "" - Math.floor((e.pageY - draggableWordHeight) / 2) + "px"
        }
    });
    $("#dragged-word-container").append(draggedWordElement);
    --appBusy;
}

function focusOnNode(node) {
    if (!graph)
        return;
    ++appBusy;
    const distance = 160;
    const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
    if (numDimensions == 3) {
        graph.cameraPosition({
            x: node.x * distRatio,
            y: node.y * distRatio,
            z: node.z * distRatio
        }, // new position
            node, // lookAt ({ x, y, z })
            2000 // ms transition duration
        );
    }
    --appBusy;
}

function focusOnWord(word) {
    if (!graph)
        return;
    var node = graphData.nodes.find(element => element.id == word);
    if (!node)
        return;
    focusOnNode(node);
}

// Create a new graph ujnder $('graph').
function initializeGraph() {
    ++appBusy;
    $("#graph").empty();
    graph = ForceGraph3D()
        (document.getElementById('graph'))
        .graphData(graphData)
        .cooldownTime(Infinity) // This is necessary for depth representation.
        .numDimensions(numDimensions)
        .showNavInfo(true)
        .enableNodeDrag(false)
        .nodeAutoColorBy('group')
        .width(width)
        .height(height)
        .nodeThreeObject(node => {
            // If there is no three.js object corresponding to the node, create one.
            // Otherwise, update the color of the existing three.js object based upon
            // the distance bewteen the camera and the node.
            var sprite = null;
            var c = graph.cameraPosition();
            if (node.hasOwnProperty('__threeObj')) {
                sprite = node.__threeObj;
            } else {
                sprite = new SpriteText(node.id);
                sprite.material.depthWrite = true;
                sprite.color = node.color;
                sprite.textHeight = 8;
            }
            if (numDimensions == 3) {
                var dist = calculateDistanceFromCamera(node.x, node.y, node.z);
                sprite.color = node.color + (Math.trunc(dist * 255)).toString(16).padStart(2, '0');
                sprite.backgroundColor = "transparent";
            } else {
                sprite.color = node.color;
            }
            if (selectedWord && node.id == selectedWord)
                sprite.color = "white";
            return sprite;
        })
        .onNodeHover((current, prev) => {
            selectedNode = current ? current : null;
            selectedWord = current ? current.id : null;

            if (selectedWord && (prev == null || selectedWord != prev.id)) {
                var prevSelectedWord = selectedWord;
                $.getJSON("https://api.dictionaryapi.dev/api/v2/entries/en/" + selectedWord, function (data) {
                    setTimeout(function () {
                        if (prevSelectedWord == selectedWord) {
                            $("#invisible-tooltip-holder").tooltip('hide');
                            $("#invisible-tooltip-holder").tooltip('dispose');
                            $("#invisible-tooltip-holder").empty();
                            $("#invisible-tooltip-holder").css('left', pageX);
                            $("#invisible-tooltip-holder").css('top', pageY + 12);
                            $('#invisible-tooltip-holder').tooltip({
                                trigger: 'manual',
                                animated: 'fade',
                                placement: 'bottom',
                                html: true,
                                title: data[0].meanings[0].definitions[0].definition,
                                container: '#wrapper'
                            });
                            $("#invisible-tooltip-holder").tooltip('show');
                        }
                    }, 200);
                });
            } else if (!selectedWord && prev) {
                $("#invisible-tooltip-holder").tooltip('hide');
                $("#invisible-tooltip-holder").tooltip('dispose');
                $("#invisible-tooltip-holder").empty();
            }
        });
    graph.onEngineTick(() => {
        const c = graph.cameraPosition();
        graph.nodeThreeObject(node => {
            var sprite = null;
            if (node.hasOwnProperty('__threeObj')) {
                sprite = node.__threeObj;
            } else {
                sprite = new SpriteText(node.id);
                sprite.material.depthWrite = true;
                sprite.color = node.color;
                sprite.textHeight = 8;
            }
            if (numDimensions == 3) {
                var dist = calculateDistanceFromCamera(c, node.x, node.y, node.z);
                sprite.color = node.color + (Math.trunc(dist * 255)).toString(16).padStart(2, '0');
                sprite.backgroundColor = "transparent";
            } else {
                sprite.color = node.color;
            }
            if (selectedWord && node.id == selectedWord) {
                sprite.color = "#000000cc";
                sprite.backgroundColor = node.color + "cc";
                sprite.text = sprite.text;
                sprite.padding = 1;
            }
            return sprite;
        });
        graph.linkColor(link => {
            if (numDimensions == 3) {
                var dist = calculateDistanceFromCamera(
                    c,
                    (link.source.x + link.target.x) / 2,
                    (link.source.y + link.target.y) / 2,
                    (link.source.z + link.target.z) / 2);
                var comp = (Math.trunc(dist * 255)).toString(16).padStart(2, '0');
                return "#" + comp + comp + comp;
            } else {
                return "#ffffffcc";
            }
        });
    });
    graph
        .d3Force('link')
        .distance(link => link.distance);
    graph.d3Force('charge').strength(chargeStrength);
    graph.resumeAnimation();
    --appBusy;
}

function processRelatedWords(keyword, result) {
    console.log(result)
    graphDataNextGroup += 1;
    $.each(result, function (i, edge) {
        var similarity = edge.distance;
        var relativeFrequency = edge['relative-frequency'];
        var isRelativeFrequencyInRange = (1 <= relativeFrequency && relativeFrequency <= 2000);
        var isSimilarityInRange = (numSetWords >= 1 && keyword == word1) ? (0.05 <= Math.abs(similarity) && Math.abs(similarity) <= 0.5) : true;
        var extendSearch = false;
        if ((numSetWords >= 2 && keyword == word1) || (numSetWords >= 3 && keyword == word2)) {
            isRelativeFrequencyInRange = isSimilarityInRange = true;
            // extendSearch = true;
        }

        if (isRelativeFrequencyInRange && isSimilarityInRange && edge.word.match(wordRegexPattern)) {
            var word = edge.word;
            var distance = 1 - edge.distance;
            distance = distance * (maxScaledDistance - minScaledDistance) + minScaledDistance;
            if (!graphData.nodes.find(element => element.id == word)) {
                graphData.nodes.push({
                    group: graphDataNextGroup,
                    id: word
                });
                if (extendSearch) searchConceptNet(word);
                else newWords.push(word);
            }
            if (!graphData.nodes.find(element => ((element.source == keyword && element.target == word) || (element.source == word && element.target == keyword)))) {
                graphData.links.push({
                    source: keyword,
                    target: word,
                    distance: distance
                });
            }
        }
    });
    graph.graphData(graphData);
    graph.refresh();
}

function searchConceptNet(keyword) {
    if (!keyword.match(wordRegexPattern))
        return;
    var keywordNode = null;

    if (!graphData.nodes.find(element => { var found = (element.id == keyword); if (found) keywordNode = element; return found; })) {
        keywordNode = {
            group: graphDataNextGroup + 1,
            id: keyword
        };
        graphData.nodes.push(keywordNode);
    }
    graph.graphData(graphData);

    if (debugMode) {
        showAlert(keyword);
        showAlert(url);
    }

    if (soundEffect_started)
        soundEffect_started.play();
    addWordToWordPool(keyword);
    var antonyms = (numSetWords >= 2 && keyword == word1) || (numSetWords >= 3 && keyword == word2);
    var cached = relatedWordsCache[keyword];
    var url = antonyms
                  ? ("https://yuriom.pythonanywhere.com/api/list-antonyms-of-adjective?word=" + keyword)
                  : ("https://yuriom.pythonanywhere.com/api/list-related-adjectives?limit=500&word=" + keyword);
    if (cached && !antonyms) {
        processRelatedWords(keyword, relatedWordsCache[keyword]);
    } else {
        $.getJSON(
            url,
            function (data) {
                if (!antonyms)
                    relatedWordsCache[keyword] = data;
                processRelatedWords(keyword, data);
            });
    }
}

function timeToString(time) {
    var diffInHrs = time / 3600000;
    var hh = Math.floor(diffInHrs);

    var diffInMin = (diffInHrs - hh) * 60;
    var mm = Math.floor(diffInMin);

    var diffInSec = (diffInMin - mm) * 60;
    var ss = Math.floor(diffInSec);

    var diffInMs = (diffInSec - ss) * 100;
    var ms = Math.floor(diffInMs);

    var formattedMM = mm.toString().padStart(2, "0");
    var formattedSS = ss.toString().padStart(2, "0");
    var formattedMS = ms.toString().padStart(2, "0");

    return `${formattedMM}:${formattedSS}:${formattedMS}`;
}

function startStopwatch() {
    timer_startTime = Date.now() - timer_elapsedTime;
    timer_interval = setInterval(function printTime() {
        timer_elapsedTime = Date.now() - timer_startTime;
        document.getElementById("stopwatch-display").innerHTML = timeToString(timer_elapsedTime);
    }, 10);
    document.getElementById("stopwatch-button").innerHTML = (inPractice ? "Finish Practice" : "Finish");
    $("#stopwatch-display").css("visibility", "hidden");
}

function stopStopwatch() {
    clearInterval(timer_interval);
    //document.getElementById("stopwatch-display").innerHTML = ("00:00:00");
    timer_elapsedTime = 0;
    document.getElementById("stopwatch-button").innerHTML = (inPractice ? "Start Practice" : "Start");
    $("#stopwatch-display").css("visibility", "visible");
}

$("#stopwatch-button").click(function () {
    var buttonText = document.getElementById("stopwatch-button").innerHTML;
    if (buttonText == "Start" || buttonText == "Start Practice") {
        if (graph)
            clearGraph();
        clearCharacterSpace();
        $("#design-brief").text(designBrief);
        $("#design-brief").css("visibility", "visible");
        startStopwatch();
    } else {
        stopStopwatch();
    }
});

function updateCharacterSpace() {
    $("#word1-container").text(numSetWords >= 1 ? word1 : "word1").css("color", numSetWords >= 1 ? "LightBlue" : "white");
    $("#word2-container").text(numSetWords >= 2 ? word2 : "word2").css("color", numSetWords >= 2 ? "LightBlue" : "white");
    $("#word3-container").text(numSetWords >= 3 ? word3 : "word3").css("color", numSetWords >= 3 ? "LightPink" : "white");
    $("#word4-container").text(numSetWords >= 4 ? word4 : "word4").css("color", numSetWords >= 4 ? "LightPink" : "white");
    // if (numSetWords >= 1) $("#word1-container").addClass("draggable-word"); else $("#word1-container").removeClass("draggable-word");
    // if (numSetWords >= 2) $("#word2-container").addClass("draggable-word"); else $("#word2-container").removeClass("draggable-word");
    // if (numSetWords >= 3) $("#word3-container").addClass("draggable-word"); else $("#word3-container").removeClass("draggable-word");
    // if (numSetWords >= 4) $("#word4-container").addClass("draggable-word"); else $("#word4-container").removeClass("draggable-word");
    $("#design-concept-map-quadrant-internal-area1").text("\"" + word1 + " " + word2 + "\"").css("background-color", numSetWords >= 2 ? "LightBlue" : "transparent");
    $("#design-concept-map-quadrant-internal-area2").text("\"" + word3 + " " + word2 + "\"").css("background-color", numSetWords >= 3 ? "LightPink" : "transparent");
    $("#design-concept-map-quadrant-internal-area3").text("\"" + word3 + " " + word4 + "\"").css("background-color", numSetWords >= 4 ? "LightPink" : "transparent");
    $("#design-concept-map-quadrant-internal-area4").text("\"" + word1 + " " + word4 + "\"").css("background-color", numSetWords >= 4 ? "LightPink" : "transparent");
}

function clearCharacterSpace() {
    numSetWords = 0;
    word1 = word2 = word3 = word4 = null;
    updateCharacterSpace();
}

function setWordInCharacterSpace(searchWord) {
    if (numSetWords < 4) {
        if (numSetWords == 0) {
            word1 = searchWord;
        } else if (numSetWords == 1) {
            if (word1 == searchWord) return;
            word2 = searchWord;
        } else if (numSetWords == 2) {
            if (word1 == searchWord) return;
            if (word2 == searchWord) return;
            word3 = searchWord;
        } else if (numSetWords == 3) {
            if (word1 == searchWord) return;
            if (word2 == searchWord) return;
            if (word3 == searchWord) return;
            word4 = searchWord;
        }
        ++numSetWords;
        updateCharacterSpace();

        if (soundEffect_wordSet)
            soundEffect_wordSet.play();
        addWordToWordPool(searchWord);

        setTimeout(function() {
            if (!withDictionary) {
                if (numSetWords == 1) {
                    showAlert("Word 1 is set to \"" + word1 + ".\" Now searching for words related to \"" + word1 + "\"...");
                    clearGraph();
                    searchConceptNet(word1);
                } else if (numSetWords == 2) {
                    showAlert("Word 2 is set to \"" + word2 + ".\" Now searching for antonyms of \"" + word1 + "\"...");
                    clearGraph();
                    searchConceptNet(word1);
                } else if (numSetWords == 3) {
                    showAlert("Word 3 is set to \"" + word3 + ".\" Now searching for antonyms of \"" + word2 + "\"...");
                    clearGraph();
                    searchConceptNet(word2);
                } else if (numSetWords) {
                    showAlert("Word 4 is set to \"" + word4 + ".\" Congratulations!");
                }
            } else {
                if (numSetWords == 1) {
                    showAlert("Word 1 is set to \"" + word1 + ".\"");
                } else if (numSetWords == 2) {
                    showAlert("Word 2 is set to \"" + word2 + ".\"");
                } else if (numSetWords == 3) {
                    showAlert("Word 3 is set to \"" + word3 + ".\"");
                } else if (numSetWords) {
                    showAlert("Word 4 is set to \"" + word4 + ".\" Congratulations!");
                }
            }
        }, 1000);
    }
}

function undoSetWord() {
    if (numSetWords == 4) {
        $("#word" + numSetWords + "-container").css("color", "white").text("word4").removeClass("draggable-word");
        $("#design-concept-map-quadrant-internal-area3").css("background-color", "transparent").text("");
        $("#design-concept-map-quadrant-internal-area4").css("background-color", "transparent").text("");
    } else if (numSetWords == 3) {
        $("#word" + numSetWords + "-container").css("color", "white").text("word3").removeClass("draggable-word");
        $("#design-concept-map-quadrant-internal-area2").css("background-color", "transparent").text("");
    } else if (numSetWords == 2) {
        $("#word" + numSetWords + "-container").css("color", "white").text("word2").removeClass("draggable-word");
        $("#dsesign-concept-map-quadrant-internal-area1").css("background-color", "transparent").text("");
    } else if (numSetWords == 1) {
        $("#word" + numSetWords + "-container").css("color", "white").text("word1").removeClass("draggable-word");
    }
    --numSetWords;
}

function addWordToWordPool(word) {
    if ($("#word-pool").has(".pooled-word--" + word).length)
        return;
    var devareButton = $("<span />", {
        text: "x",
        css: {
            "color": "red",
            "background-color": "transparent"
        }
    }).click(function () {
        $(this).parent().remove();
        return false;
    });
    var draggableWord = $("<span />", {
        class: "draggable-word pooled-word--" + word,
        text: word,
        css: {
            "color": "white",
            "background-color": "transparent"
        }
    });
    var newEntry = $("<span />").append(draggableWord).append(devareButton);
    $("#word-pool").append(newEntry);
}

function isWordAlreadyInCharacterSpace(word) {
    return (numSetWords >= 1 && word == word1) || (numSetWords >= 2 && word == word2) || (numSetWords >= 3 && word == word3) || (numSetWords >= 4 && word == word4);
}

function lookUpWordInDictionary(word) {
    if (!withDictionary)
        return;
    $("#graph").empty();
    $("#graph").append($("<iframe />", {
        id: "online-dictionary-content",
        width: width,
        height: height,
        src: "https://www.merriam-webster.com/thesaurus/" + word,
        css: {
            "border": "0px"
        }
    }));
    addWordToWordPool(word);
}

function wordFilter(word) {
    return word.match(wordRegexPattern);
}

function adjectiveChecker(word, onSuccess)
{
    $.getJSON("https://yuriom.pythonanywhere.com/api/is-adjective?word=" + word,
        function (data) {
            if (data.result) {
                onSuccess(data.word);
            } else {
                showAlert("Please enter an adjective.", "danger");
            }
        });
}

function defineTextBoxWithButton(textBoxSelector, buttonSelector, filter, handler)
{
    $(textBoxSelector).on("keypress", function (e) {
        var word = $(textBoxSelector).val().trim();
        const key = e.keyCode || e.charCode || 0;
        if (key == 13 && filter(word)) {
            e.preventDefault();
            handler(word);
            $(textBoxSelector).val("");
        }
    });

    $(buttonSelector).on("click", function (e) {
        var word = $(textBoxSelector).val().trim();
        if (filter(word)) {
            handler(word);
            $(textBoxSelector).val("");
        }
    });
}

function init() {

    defineTextBoxWithButton("#character-space-box", "#set-word-in-character-space-button", wordFilter, setWordInCharacterSpace);
    defineTextBoxWithButton("#word-pool-box", "#add-word-to-word-pool-button", wordFilter, addWordToWordPool);

    var pointerDownHandler = (e) => {
        if (!e.isTrusted)
            return;
        pageX = e.pageX;
        pageY = e.pageY;
        var leftMouseButtonOnlyDown = e.buttons === undefined ? e.which === 1 : e.buttons === 1;
        if (leftMouseButtonOnlyDown && !draggedWord && (($("#graph").has($(e.target)).length && selectedWord) || $(e.target).hasClass("draggable-word")) && !draggedWordElement) {
            if ($(e.target).hasClass("draggable-word")) {
                draggedWord = $(e.target).text();
                createDraggedWord(e, draggedWord);
            } else {
                draggedWord = selectedWord;
            }
            e.stopPropagation();
            dragMoved = false;
            dragInitialX = e.pageX;
            dragInitialY = e.pageY;
            if (!$("#graph").has($(e.target)).length) {
                focusOnWord(draggedWord);
                lookUpWordInDictionary(draggedWord);
            }
        } else {
            draggedWord = null;
        }
    };
    var pointerMoveHandler = (e) => {
        pageX = e.pageX;
        pageY = e.pageY;
        if (draggedWord) {
            e.stopPropagation();
            if (!draggedWordElement) {
                createDraggedWord(e, draggedWord);
            }
            if (e.pageX != dragInitialX || e.pageY != dragInitialY)
                dragMoved = true;
            draggedWordElement.css("left", "" + (e.pageX - Math.floor(draggableWordWidth / 2)) + "px");
            draggedWordElement.css("top", "" + (e.pageY - Math.floor(draggableWordHeight / 2)) + "px");
        }
    };
    var pointerUpHandler = (e) => {
        pageX = e.pageX;
        pageY = e.pageY;
        if (draggedWord) {
            var word = draggedWord;
            $("#dragged-word-container").empty();
            draggedWord = null;
            draggedWordElement = null;
            var el = document.elementFromPoint(e.pageX, e.pageY);
            console.log(graphData);
            // console.log(el);
            // drop
            if ($("#graph").has($(el)).length) {
                if (graph)
                    focusOnWord(word);
                searchConceptNet(word);

            } else if (($(el).attr('id') == "word1-container" || $("#word1-container").has($(el)).length) && numSetWords >= 4 && !isWordAlreadyInCharacterSpace(word)) {
                word1 = word;
                updateCharacterSpace();
                if (withDictionary) {
                    showAlert("Word 1 is set to \"" + word1 + ".\"");
                } else {
                    showAlert("Word 1 is set to \"" + word1 + ".\" Now searching for words related to \"" + word1 + "\"...");
                    if (graph)
                        clearGraph();
                    searchConceptNet(word1);
                }
            } else if (($(el).attr('id') == "word2-container" || $("#word2-container").has($(el)).length) && numSetWords >= 4 && !isWordAlreadyInCharacterSpace(word)) {
                word2 = word;
                updateCharacterSpace();
                if (withDictionary) {
                    showAlert("Word 2 is set to \"" + word2 + ".\"");
                } else {
                    showAlert("Word 2 is set to \"" + word2 + ".\" Now searching for antonyms of \"" + word1 + "\"...");
                    if (graph)
                        clearGraph();
                    searchConceptNet(word1);
                }
            } else if (($(el).attr('id') == "word3-container" || $("#word3-container").has($(el)).length) && numSetWords >= 4 && !isWordAlreadyInCharacterSpace(word)) {
                word3 = word;
                updateCharacterSpace();
                if (withDictionary) {
                    showAlert("Word 3 is set to \"" + word3 + ".\"");
                } else {
                    showAlert("Word 3 is set to \"" + word3 + ".\" Now searching for antonyms of \"" + word2 + "\"...");
                    if (graph)
                        clearGraph();
                    searchConceptNet(word2);
                }
            } else if (($(el).attr('id') == "word4-container" || $("#word4-container").has($(el)).length) && numSetWords >= 4 && !isWordAlreadyInCharacterSpace(word)) {
                word4 = word;
                if (withDictionary) {
                    showAlert("Word 4 is set to \"" + word4 + ".\"");
                } else {
                    // No need to clear the graph.
                    updateCharacterSpace();
                    showAlert("Word 4 is set to \"" + word4 + ".\"");
                }
            } else if ($("#design-concept-map").has($(el)).length && numSetWords < 4) {
                setWordInCharacterSpace(word);
            } else if ($(".heading").has($(el)).length) {
            } else if ($("#control-panel").has($(el)).length) {
                addWordToWordPool(word);
            } else if ($("#lower-pane").has($(el)).length) {
                addWordToWordPool(word);
            }
        }
    };
    var options = { capture: true };
    document.addEventListener("pointerdown", pointerDownHandler, options);
    document.addEventListener("pointermove", pointerMoveHandler, options);
    document.addEventListener("pointerup", pointerUpHandler, options);

    $("#undo-set-word-button").on("click", function (e) {
        undoSetWord();
    });

    if (withDictionary) {
        $("#clear-button").on("click", function () {
            $("#graph").empty();
        });

        defineTextBoxWithButton("#search-box", "#search-button", wordFilter, function(word) {
           adjectiveChecker(word, lookUpWordInDictionary);
        });
    } else {
        setComponentSizes();
        initializeGraph();

        $("#graph-2d-button").on("click", function () {
            if (numDimensions == 3) {
                toggleDimention();
            }
        });

        $("#graph-3d-button").on("click", function () {
            if (numDimensions == 2) {
                toggleDimention();
            }
        });

        $("#zoom-to-fit-button").on("click", function () {
            console.log(graphData);
            if (graph)
                graph.zoomToFit(2000, 10, () => true);
        });

        $("#debug-button").on("click", function () {
            $("#debug-button").toggleClass("btn-outline-secondary");
            $("#debug-button").toggleClass("btn-secondary");
            debugMode = !debugMode;
        });

        $("#clear-button").on("click", function () {
            graphData = {
                nodes: [],
                links: []
            };
            graph.graphData(graphData);
        });

        defineTextBoxWithButton("#search-box", "#search-button", wordFilter, function(word) {
           adjectiveChecker(word, searchConceptNet);
        });

        setInterval(function () {
            if (appBusy)
                return;
            if (!isCacheBeingLoaded && newWords.length > 0) {
                isCacheBeingLoaded = true;
                var word = newWords.pop();
                $.getJSON("https://yuriom.pythonanywhere.com/api/list-related-adjectives?word=" + word,
                    function (data) {
                        (relatedWordsCache[word]) = data;
                        isCacheBeingLoaded = false;
                    });
            }
        }, 100);
    }

    window.onhashchange = function () {
        window.location.reload();
    };

    $("body").css("visibility", "visible");
    $("body").css("pointer-events", "auto");

    setInterval(function () {
        setComponentSizes();
    }, 100);

    --appBusy;
}
$(document).ready(init);
