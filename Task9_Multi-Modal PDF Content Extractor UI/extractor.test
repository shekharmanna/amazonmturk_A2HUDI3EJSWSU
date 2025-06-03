// extractor.test.js

/**
 * Jest test suite for the Multi-Modal PDF Content Extractor UI.
 *
 * This suite focuses on testing the client-side JavaScript logic,
 * including DOM manipulation, event handling, and analysis functions.
 * External libraries like PDF.js are mocked to allow for isolated unit testing
 * without requiring actual PDF parsing in the Node.js environment.
 */

// Mock PDF.js library
// We need to mock the entire pdfjsLib object and its methods that are called.
// This allows us to control the behavior of PDF loading and content extraction
// without actually loading a PDF in the test environment.
const mockGetPage = jest.fn();
const mockGetTextContent = jest.fn();
const mockGetMetadata = jest.fn();
const mockRender = jest.fn();
const mockGetOperatorList = jest.fn();
const mockGetViewport = jest.fn();

// Mock the getDocument function to return a mock PDF object
const mockGetDocument = jest.fn(() => ({
    promise: Promise.resolve({
        numPages: 1, // Simulate a single-page PDF for simplicity
        getPage: mockGetPage.mockImplementation((pageNum) => {
            // Mock the page object returned by getPage
            return Promise.resolve({
                getTextContent: mockGetTextContent.mockResolvedValue({
                    items: [{ str: `Sample text from page ${pageNum}.` }]
                }),
                getMetadata: mockGetMetadata.mockResolvedValue({ info: { Title: 'Mock PDF', Author: 'Jest Test' } }),
                render: mockRender.mockImplementation(() => ({ promise: Promise.resolve() })),
                getOperatorList: mockGetOperatorList.mockResolvedValue({}), // Mock empty operator list for images
                getViewport: mockGetViewport.mockReturnValue({ width: 800, height: 600 })
            });
        }),
        getMetadata: mockGetMetadata.mockResolvedValue({ // Mock metadata for the document itself
            info: {
                Title: 'Test Document Title',
                Author: 'Test Author',
                Creator: 'Test Creator',
                Producer: 'Test Producer',
                CreationDate: 'D:20230101000000Z',
                ModDate: 'D:20230101000000Z',
                PDFFormatVersion: '1.7'
            }
        })
    })
}));

// Mock the global pdfjsLib object
global.pdfjsLib = {
    getDocument: mockGetDocument,
    GlobalWorkerOptions: {
        workerSrc: '' // Mock workerSrc as it's not relevant for JSDOM tests
    }
};

// Mock Mammoth.js (if it were used, but not directly in this PDF flow)
global.mammoth = {
    convertToHtml: jest.fn().mockResolvedValue({ value: '<p>Mock DOCX content</p>' })
};


// Load the HTML content into JSDOM before each test suite
// This simulates the browser environment for our tests.
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.resolve(__dirname, './index.html'), 'utf8');

// Describe block for the entire test suite
describe('Multi-Modal PDF Content Extractor UI', () => {
    // Before each test, reset the DOM and re-insert the HTML.
    // This ensures a clean slate for every test, preventing side effects.
    beforeEach(() => {
        document.documentElement.innerHTML = html;
        // Re-run the script from the HTML to re-initialize event listeners and global variables
        // This is crucial because JSDOM doesn't automatically re-execute <script> tags on innerHTML change.
        const scriptContent = document.querySelector('script:not([src])').textContent;
        eval(scriptContent); // eslint-disable-line no-eval

        // Reset mocks for each test to ensure isolation
        jest.clearAllMocks();
        // Re-mock specific PDF.js functions if they were changed
        global.pdfjsLib.getDocument = mockGetDocument;
        global.pdfjsLib.GlobalWorkerOptions = { workerSrc: '' };
    });

    // Test Case 1: Initial UI state
    test('should display initial UI elements correctly', () => {
        expect(document.querySelector('h1').textContent).toContain('Multi-Modal PDF Content Extractor');
        expect(document.getElementById('fileInput')).toBeInTheDocument();
        expect(document.getElementById('extractBtn')).toBeDisabled();
        expect(document.getElementById('resultsSection')).not.toBeVisible();
        expect(document.getElementById('progressBar')).not.toBeVisible();
        expect(document.getElementById('statusMessage')).not.toBeVisible();
    });

    // Test Case 2: File selection enables extract button and shows file info
    test('should enable extract button and show file info after file selection', () => {
        const testFile = new File(['dummy pdf content'], 'test.pdf', { type: 'application/pdf' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(testFile);

        // Simulate file input change event
        const fileInput = document.getElementById('fileInput');
        fileInput.files = dataTransfer.files;
        fileInput.dispatchEvent(new Event('change', { bubbles: true }));

        expect(document.getElementById('extractBtn')).toBeEnabled();
        expect(document.getElementById('fileInfo')).toBeVisible();
        expect(document.getElementById('fileList').textContent).toContain('test.pdf');
    });

    // Test Case 3: Drag and drop PDF file
    test('should handle drag and drop of a PDF file', () => {
        const testFile = new File(['dummy pdf content'], 'dragged.pdf', { type: 'application/pdf' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(testFile);

        const uploadArea = document.getElementById('uploadArea');

        // Simulate dragover, drop, and dragleave events
        uploadArea.dispatchEvent(new DragEvent('dragover', { dataTransfer, bubbles: true }));
        expect(uploadArea).toHaveClass('dragover');

        uploadArea.dispatchEvent(new DragEvent('drop', { dataTransfer, bubbles: true }));
        expect(uploadArea).not.toHaveClass('dragover'); // Should remove dragover class on drop

        expect(document.getElementById('extractBtn')).toBeEnabled();
        expect(document.getElementById('fileInfo')).toBeVisible();
        expect(document.getElementById('fileList').textContent).toContain('dragged.pdf');
    });

    // Test Case 4: Drag and drop non-PDF file (should show error)
    test('should show error for drag and drop of non-PDF file', () => {
        const testFile = new File(['dummy text content'], 'image.jpg', { type: 'image/jpeg' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(testFile);

        const uploadArea = document.getElementById('uploadArea');
        uploadArea.dispatchEvent(new DragEvent('drop', { dataTransfer, bubbles: true }));

        expect(document.getElementById('extractBtn')).toBeDisabled(); // Button should remain disabled
        expect(document.getElementById('statusMessage')).toBeVisible();
        expect(document.getElementById('statusMessage').textContent).toContain('Only PDF files are supported for drag & drop.');
        expect(document.getElementById('statusMessage')).toHaveClass('error-message');
    });

    // Test Case 5: Tab switching functionality
    test('should switch active tab content on click', () => {
        const textTab = document.querySelector('.tab[data-tab="text"]');
        const imagesTab = document.querySelector('.tab[data-tab="images"]');
        const textContent = document.getElementById('textContent');
        const imagesContent = document.getElementById('imagesContent');

        // Initially text tab is active
        expect(textTab).toHaveClass('active');
        expect(textContent).toHaveClass('active');
        expect(imagesTab).not.toHaveClass('active');
        expect(imagesContent).not.toHaveClass('active');

        // Click images tab
        imagesTab.click();

        expect(imagesTab).toHaveClass('active');
        expect(imagesContent).toHaveClass('active');
        expect(textTab).not.toHaveClass('active');
        expect(textContent).not.toHaveClass('active');
    });

    // Test Case 6: Clear results button functionality
    test('should clear results and reset UI on clear button click', () => {
        // Simulate some data and UI state
        extractedData.text = 'Some extracted text';
        extractedData.images = [{ page: 1, data: 'data:image/png;base64,...', width: 100, height: 100 }];
        document.getElementById('extractedText').textContent = 'Some text';
        document.getElementById('extractedImages').innerHTML = '<img src="test.png">';
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('extractBtn').disabled = false;
        document.getElementById('fileInfo').style.display = 'block';

        document.getElementById('clearBtn').click();

        expect(extractedData.text).toBe('');
        expect(extractedData.images).toEqual([]);
        expect(document.getElementById('extractedText').textContent).toBe('');
        expect(document.getElementById('extractedImages').innerHTML).toBe('');
        expect(document.getElementById('resultsSection')).not.toBeVisible();
        expect(document.getElementById('extractBtn')).toBeDisabled();
        expect(document.getElementById('fileInfo')).not.toBeVisible();
        expect(document.getElementById('fileList').innerHTML).toBe('');
        expect(document.getElementById('statusMessage').textContent).toBe(''); // Should clear status
    });

    // Test Case 7: PDF extraction flow (mocked)
    test('should process PDF and display results when extract button is clicked', async () => {
        const testFile = new File(['dummy pdf content'], 'sample.pdf', { type: 'application/pdf' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(testFile);
        document.getElementById('fileInput').files = dataTransfer.files;
        document.getElementById('fileInput').dispatchEvent(new Event('change', { bubbles: true }));

        // Mock PDF.js internal calls for this specific test
        mockGetTextContent.mockResolvedValueOnce({ items: [{ str: 'Hello PDF text.' }] });
        mockGetMetadata.mockResolvedValueOnce({
            info: { Title: 'Mocked Doc', Author: 'Mock Author', CreationDate: 'D:20230101000000Z' }
        });
        mockRender.mockImplementationOnce(() => ({ promise: Promise.resolve() })); // For image extraction

        document.getElementById('extractBtn').click();

        // Wait for async operations to complete
        await Promise.resolve(); // Allow microtasks to run (for initial promise resolution)
        await new Promise(resolve => setTimeout(resolve, 100)); // Small delay for UI updates/async loops

        expect(document.getElementById('progressBar')).toBeVisible();
        expect(document.getElementById('statusMessage')).toBeVisible();
        expect(document.getElementById('statusMessage').textContent).toContain('Processing sample.pdf...');

        // Simulate completion of extraction loop
        await new Promise(resolve => setTimeout(resolve, 500)); // Give time for the loop to finish

        expect(mockGetDocument).toHaveBeenCalledWith({ data: expect.any(ArrayBuffer) });
        expect(mockGetPage).toHaveBeenCalledWith(1);
        expect(mockGetTextContent).toHaveBeenCalled();
        expect(mockGetMetadata).toHaveBeenCalled();
        expect(mockRender).toHaveBeenCalled(); // Should be called for image extraction

        expect(document.getElementById('resultsSection')).toBeVisible();
        expect(document.getElementById('extractedText').textContent).toContain('Hello PDF text.');
        expect(document.getElementById('metadataTable').innerHTML).toContain('Test Document Title'); // From initial mock
        expect(document.getElementById('extractedImages').innerHTML).toContain('data:image/png'); // Should contain image data URL
        expect(document.getElementById('statusMessage').textContent).toContain('Extraction completed successfully!');
        expect(document.getElementById('statusMessage')).toHaveClass('success-message');
    });

    // Test Case 8: Error during PDF extraction
    test('should show error message if PDF extraction fails', async () => {
        const testFile = new File(['corrupt pdf'], 'corrupt.pdf', { type: 'application/pdf' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(testFile);
        document.getElementById('fileInput').files = dataTransfer.files;
        document.getElementById('fileInput').dispatchEvent(new Event('change', { bubbles: true }));

        // Mock getDocument to throw an error
        mockGetDocument.mockImplementationOnce(() => ({
            promise: Promise.reject(new Error('PDF parsing error'))
        }));

        document.getElementById('extractBtn').click();

        await new Promise(resolve => setTimeout(resolve, 100)); // Allow async operations to start
        await new Promise(resolve => setTimeout(resolve, 500)); // Allow promise rejection to propagate

        expect(document.getElementById('statusMessage')).toBeVisible();
        expect(document.getElementById('statusMessage').textContent).toContain('Extraction failed: PDF parsing error');
        expect(document.getElementById('statusMessage')).toHaveClass('error-message');
        expect(document.getElementById('progressBar')).not.toBeVisible();
        expect(document.getElementById('extractBtn')).toBeEnabled(); // Should re-enable button
    });

    // --- Analysis Function Tests ---

    // Test Case 9: analyzeStructure
    test('analyzeStructure should correctly count lines, paragraphs, sentences, words, and characters', () => {
        const text = "This is sentence one. This is sentence two.\n\nThis is a new paragraph. It has more words.";
        const result = analyzeStructure(text); // Assuming analyzeStructure is globally available due to eval(scriptContent)

        expect(result.lines).toBe(3); // "This is sentence one. This is sentence two.", "", "This is a new paragraph. It has more words."
        expect(result.paragraphs).toBe(2);
        expect(result.sentences).toBe(4);
        expect(result.words).toBe(19);
        expect(result.characters).toBe(text.length);
        expect(result.avgWordsPerSentence).toBe(5); // 19 words / 4 sentences = 4.75 -> 5
        expect(result.avgSentencesPerParagraph).toBe(2); // 4 sentences / 2 paragraphs = 2
    });

    // Test Case 10: extractKeywords
    test('extractKeywords should extract relevant keywords and filter common words', () => {
        const text = "The quick brown fox jumps over the lazy dog. Dogs are loyal animals. Fox and dog are common words.";
        const result = extractKeywords(text);

        // Expect keywords to be sorted by frequency, common words filtered
        // 'dog' appears twice, 'fox' twice, 'jumps' once, 'lazy' once, 'quick' once, 'brown' once
        // Common words: 'the', 'are', 'and'
        const expectedKeywords = [
            { word: 'dog', count: 2 },
            { word: 'fox', count: 2 },
            { word: 'jumps', count: 1 },
            { word: 'lazy', count: 1 },
            { word: 'quick', count: 1 },
            { word: 'brown', count: 1 },
            { word: 'dogs', count: 1 },
            { word: 'loyal', count: 1 },
            { word: 'animals', count: 1 },
            { word: 'common', count: 1 },
        ].sort((a,b) => b.count - a.count || a.word.localeCompare(b.word)); // Sort for consistent order

        // Filter out keywords with count 1 that are not in the expected list
        const actualKeywords = result.filter(k => k.count > 1 || expectedKeywords.some(ek => ek.word === k.word && ek.count === k.count))
                                     .sort((a,b) => b.count - a.count || a.word.localeCompare(b.word));

        expect(actualKeywords).toEqual(expect.arrayContaining([
            { word: 'dog', count: 2 },
            { word: 'fox', count: 2 }
        ]));
        // Check for absence of common words
        expect(actualKeywords.some(k => k.word === 'the')).toBeFalsy();
        expect(actualKeywords.some(k => k.word === 'are')).toBeFalsy();
    });
    
    // Test Case 11: detectLanguage - English
    test('detectLanguage should correctly detect English', () => {
        const text = "This is a sample text in English. It contains common English words like 'the' and 'and'.";
        const result = detectLanguage(text);
        expect(result.language).toBe('English');
        expect(parseFloat(result.confidence)).toBeGreaterThan(0.5); // Confidence should be reasonably high
    });

    // Test Case 12: detectLanguage - Spanish
    test('detectLanguage should correctly detect Spanish', () => {
        const text = "Este es un texto de ejemplo en español. Contiene palabras comunes como 'que' y 'con'.";
        const result = detectLanguage(text);
        expect(result.language).toBe('Spanish');
        expect(parseFloat(result.confidence)).toBeGreaterThan(0.5);
    });

    // Test Case 13: analyzeReadability and countSyllables
    test('analyzeReadability and countSyllables should provide reasonable metrics', () => {
        const text = "The quick brown fox jumps over the lazy dog. This is a very simple sentence.";
        const result = analyzeReadability(text);

        expect(result.fleschScore).toBeCloseTo(90, -1); // Expect a high score for simple text
        expect(result.readingLevel).toBe('Very Easy (5th grade)');
        expect(result.avgWordsPerSentence).toBeGreaterThan(0);
        expect(result.avgSyllablesPerWord).toBeGreaterThan(0);

        // Test countSyllables directly
        expect(countSyllables('apple')).toBe(2); // ap-ple
        expect(countSyllables('banana')).toBe(3); // ba-na-na
        expect(countSyllables('beautiful')).toBe(3); // beau-ti-ful (simple algo)
        expect(countSyllables('love')).toBe(1); // love (silent e rule)
        expect(countSyllables('table')).toBe(2); // ta-ble
    });

    // Test Case 14: Empty text for analysis functions
    test('analysis functions should handle empty text gracefully', () => {
        expect(analyzeStructure('')).toEqual({
            lines: 0, paragraphs: 0, sentences: 0, words: 0, characters: 0,
            avgWordsPerSentence: 0, avgSentencesPerParagraph: 0
        });
        expect(extractKeywords('')).toEqual([]);
        expect(detectLanguage('')).toEqual({ language: 'Unknown', confidence: '0.00' });
        expect(analyzeReadability('')).toEqual({
            fleschScore: 'N/A', readingLevel: 'N/A', avgWordsPerSentence: 'N/A', avgSyllablesPerWord: 'N/A'
        });
    });

    // Test Case 15: Initial particles creation
    test('createParticles should add particles to the DOM', () => {
        // Clear existing particles to test creation from scratch
        document.getElementById('particles').innerHTML = '';
        createParticles(); // Call the function directly
        expect(document.getElementById('particles').children.length).toBe(50);
        expect(document.querySelector('.particle')).toBeInTheDocument();
    });

    // Test Case 16: Checkbox options influence extractedData.analysis
    test('checkbox options should influence which analysis is performed', async () => {
        const testFile = new File(['dummy text'], 'test.pdf', { type: 'application/pdf' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(testFile);
        document.getElementById('fileInput').files = dataTransfer.files;
        document.getElementById('fileInput').dispatchEvent(new Event('change', { bubbles: true }));

        // Uncheck all analysis options
        document.getElementById('structureAnalysis').checked = false;
        document.getElementById('keywordExtraction').checked = false;
        document.getElementById('languageDetection').checked = false;
        document.getElementById('readabilityAnalysis').checked = false;

        mockGetTextContent.mockResolvedValueOnce({ items: [{ str: 'Short text for analysis.' }] });
        mockGetMetadata.mockResolvedValueOnce({ info: {} });
        mockRender.mockImplementationOnce(() => ({ promise: Promise.resolve() }));

        document.getElementById('extractBtn').click();
        await new Promise(resolve => setTimeout(resolve, 500)); // Allow async operations to complete

        // Only tablesNote should be present if extractTables is checked, otherwise analysis should be empty
        const expectedAnalysisKeys = document.getElementById('extractTables').checked ? ['tablesNote'] : [];
        expect(Object.keys(extractedData.analysis)).toEqual(expect.arrayContaining(expectedAnalysisKeys));
        expect(Object.keys(extractedData.analysis).length).toBe(expectedAnalysisKeys.length);

        // Check one analysis option and re-run
        document.getElementById('keywordExtraction').checked = true;
        document.getElementById('extractBtn').click();
        await new Promise(resolve => setTimeout(resolve, 500));

        expect(extractedData.analysis).toHaveProperty('keywords');
        expect(Object.keys(extractedData.analysis)).toEqual(expect.arrayContaining([...expectedAnalysisKeys, 'keywords']));
    });

});

// Custom Jest matchers for better readability (optional but good practice)
// To make these available, you might need to configure Jest's setupFilesAfterEnv
// in package.json or jest.config.js, or just include them directly in the test file.
expect.extend({
    toBeVisible(received) {
        const pass = received.style.display !== 'none' && received.offsetWidth > 0 && received.offsetHeight > 0;
        if (pass) {
            return {
                message: () => `expected ${received.tagName} to not be visible`,
                pass: true,
            };
        } else {
            return {
                message: () => `expected ${received.tagName} to be visible`,
                pass: false,
            };
        }
    },
});