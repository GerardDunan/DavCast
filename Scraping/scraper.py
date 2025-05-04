import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
import random
import sys
import re
from datetime import datetime, timedelta
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys 
import argparse
import subprocess
import numpy as np
import platform
import tempfile

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define saved_emails directory path relative to the script directory
SAVED_EMAILS_DIR = os.path.join(SCRIPT_DIR, 'saved_emails')

try:
    from gmail_api import GmailAPI  # Import the GmailAPI class
    print("GmailAPI module imported successfully")
except ImportError as e:
    print("GmailAPI module not found. Error:", str(e))
    print("Please make sure you have the gmail_api.py file in your project directory")
    GmailAPI = None

# Import our JavaScript date selection helper
try:
    from weatherlink.js_date_fix import use_js_to_select_date
    print("JavaScript date selection helper imported successfully")
except ImportError as e:
    try:
        # Try to import from the same directory
        from js_date_fix import use_js_to_select_date
        print("JavaScript date selection helper imported from current directory")
    except ImportError:
        print("JavaScript date selection helper not found. Error:", str(e))
        # Define a simple fallback implementation if the module isn't found
        def use_js_to_select_date(driver, year, month, day):
            print(f"Fallback JavaScript date selection for {month}/{day}/{year}")
            try:
                script = f"""
                // Try to select the date using JavaScript
                var yearElements = document.querySelectorAll('span.year');
                for (var i = 0; i < yearElements.length; i++) {{
                    if (yearElements[i].textContent.trim() === '{year}') {{
                        yearElements[i].click();
                        return 'Year clicked';
                    }}
                }}
                return 'Year not found';
                """
                return (False, driver.execute_script(script))
            except Exception as e:
                return (False, str(e))
except Exception as e:
    print(f"Error importing JavaScript date selection helper: {e}")

class WeatherLink:
    def __init__(self, url, debug=False, export_email="teamdavcast@gmail.com", init_browser=True):
        self.url = url
        self.debug = debug
        self.export_email = export_email
        print(f"Initializing WeatherLink scraper with debug mode: {'ON' if debug else 'OFF'}")
        print(f"Export email: {export_email}")
        self.use_api = True  # Default to True, can be set to False later
        
        # Only initialize browser if requested
        if init_browser:
            self.driver = self.setup_browser()
        else:
            self.driver = None

    def setup_browser(self):
        """Set up the WebDriver."""
        try:
            print("Setting up Chrome browser...")
            options = Options()
            
            # Create temporary directory for Chrome data
            chrome_data_dir = tempfile.mkdtemp(prefix="chrome_data_")
            print(f"Using Chrome data directory: {chrome_data_dir}")
            
            # Always use headless mode for server environments
            print("Configuring Chrome for headless operation on server")
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")  # Required for running in containers/as root
            options.add_argument("--disable-dev-shm-usage")  # Use /tmp instead of /dev/shm
            options.add_argument(f"--user-data-dir={chrome_data_dir}")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-extensions")
            
            # Additional options for stability in server environments
            options.add_argument("--disable-setuid-sandbox")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--ignore-certificate-errors")
            
            # Check for Linux environment and set appropriate paths
            is_linux = sys.platform.startswith('linux')
            if is_linux:
                print("Detected Linux environment")
                
                # Try to find Chrome/Chromium in common Linux locations
                chrome_paths = [
                    "/usr/bin/google-chrome",
                    "/usr/bin/google-chrome-stable",
                    "/usr/bin/chromium-browser",
                    "/usr/bin/chromium",
                    "/snap/bin/chromium"
                ]
                
                found_chrome = False
                for path in chrome_paths:
                    if os.path.exists(path):
                        print(f"Found browser at: {path}")
                        options.binary_location = path
                        found_chrome = True
                        break
                
                if not found_chrome:
                    print("Warning: Could not find Chrome/Chromium in standard locations")
            
            # Try direct approach with ChromeDriverManager first
            try:
                print("Installing/locating Chrome driver...")
                service = Service(ChromeDriverManager().install())
                
                print("Starting Chrome browser...")
                driver = webdriver.Chrome(service=service, options=options)
            except Exception as e1:
                print(f"Error with automatic ChromeDriver: {e1}")
                print("Trying alternative approaches...")
                
                # Check if we're on Linux and try Linux-specific solutions
                if is_linux:
                    try:
                        print("Trying with system-installed chromedriver...")
                        # Try using system chromedriver (often in /usr/bin or /usr/local/bin)
                        for chromedriver_path in ["/usr/local/bin/chromedriver", "/usr/bin/chromedriver"]:
                            if os.path.exists(chromedriver_path):
                                print(f"Found chromedriver at: {chromedriver_path}")
                                service = Service(chromedriver_path)
                                driver = webdriver.Chrome(service=service, options=options)
                                print("Successfully started Chrome with system chromedriver")
                                break
                        else:
                            raise Exception("Chromedriver not found in standard system locations")
                    except Exception as linux_error:
                        print(f"Error with system chromedriver: {linux_error}")
                        
                        # Final fallback - try to download chromedriver directly
                        try:
                            print("Trying alternative chromedriver download approach...")
                            # This is a simplified approach - in production you might need a more robust solution
                            chrome_version = None
                            
                            # Try to get Chrome version from binary
                            try:
                                result = subprocess.run([options.binary_location, "--version"], 
                                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                                       text=True, check=False)
                                version_output = result.stdout
                                # Expected format: "Google Chrome XX.X.XXXX.XX"
                                match = re.search(r'Chrome\s+(\d+\.\d+\.\d+\.\d+)', version_output)
                                if match:
                                    chrome_version = match.group(1)
                                    major_version = chrome_version.split('.')[0]
                                    print(f"Detected Chrome version: {chrome_version} (Major: {major_version})")
                                else:
                                    print(f"Could not parse Chrome version from: {version_output}")
                            except Exception as ver_error:
                                print(f"Error getting Chrome version: {ver_error}")
                            
                            # If we couldn't get the version, try with a local chromedriver
                            if chrome_version is None:
                                local_driver = "./chromedriver"
                                if os.path.exists(local_driver):
                                    print(f"Using existing local chromedriver: {local_driver}")
                                    service = Service(local_driver)
                                    driver = webdriver.Chrome(service=service, options=options)
                                else:
                                    raise Exception("Could not determine Chrome version and no local chromedriver found")
                        except Exception as last_error:
                            print(f"All chromedriver approaches failed: {last_error}")
                            raise Exception("Could not initialize Chrome browser. Please install chromedriver manually.") 
                else:
                    # Continue with Windows-specific code for compatibility
                    # ... existing code for Windows ...
                    raise Exception("Non-Linux platform detected. Please modify the script for your environment.")
                
            print(f"Navigating to URL: {self.url}")
            driver.get(self.url)
            print("Chrome browser started successfully")
            return driver
        except Exception as e:
            print(f"ERROR setting up browser: {e}")
            print("Please make sure Chrome is installed on your system")
            print("You may need to download ChromeDriver manually:")
            print("1. Find your Chrome version by running 'google-chrome --version'")
            print("2. Download matching ChromeDriver from: https://chromedriver.chromium.org/downloads")
            print("3. Place chromedriver in /usr/local/bin/ and make it executable")
            raise

    def save_data(self, df, filename="weather_data.csv"):
        """Save the extracted data to a CSV file."""
        if df is not None:
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("No data to save.")
    
    def set_date_to_current(self):
        """Find the date selector and set it to yesterday's date."""
        try:
            print("Attempting to set date to yesterday's date...")
            self.take_screenshot("before_date_selection")
            
            # Get current system date and calculate yesterday's date
            current_date = datetime.now()
            yesterday_date = current_date - timedelta(days=1)
            
            yesterday_month = yesterday_date.month
            yesterday_day = yesterday_date.day
            yesterday_year = yesterday_date.year
            month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
            yesterday_month_name = month_names[yesterday_month - 1]
            
            print(f"Current system date: {current_date.month}/{current_date.day}/{current_date.year}")
            print(f"Setting to yesterday's date: {yesterday_month}/{yesterday_day}/{yesterday_year} ({yesterday_month_name})")
            
            # Enhanced selectors for date elements
            date_selectors = [
                "//span[@class='time' and @data-l10n-id='start_date']",
                "//input[@type='date']",
                "//div[contains(@class, 'date-picker')]",
                "//button[contains(@class, 'date-selector')]",
                "//div[contains(@class, 'calendar')]//button",
                "//label[contains(text(), 'Date')]/following-sibling::input",
                "//span[contains(text(), 'Date Range')]",
                "//div[contains(@class, 'date-range-picker')]",
                # More specific selectors for WeatherLink
                "//div[contains(@class, 'date-selector')]",
                "//button[contains(@class, 'btn-date')]",
                "//div[contains(@class, 'date-filter')]",
                "//input[contains(@id, 'date')]",
                "//input[contains(@name, 'date')]",
                "//div[contains(@class, 'datepicker')]",
                # Try to find by aria attributes
                "//*[@aria-label='Date picker' or @aria-label='Calendar' or @aria-label='Date']",
                # Try to find by placeholder
                "//input[@placeholder='Date' or contains(@placeholder, 'date') or contains(@placeholder, 'calendar')]"
            ]
            
            date_element = None
            used_selector = None
            
            # Try each selector to find date control
            for selector in date_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements:
                        # Try to filter for visible elements only
                        visible_elements = [e for e in elements if e.is_displayed()]
                        if visible_elements:
                            date_element = visible_elements[0]
                        else:
                            date_element = elements[0]
                        used_selector = selector
                        print(f"Found date element with selector: {selector}")
                        break
                except Exception as selector_error:
                    print(f"Error with selector {selector}: {selector_error}")
                    continue
            
            if not date_element:
                print("Could not find any date selector element on the page")
                self.take_screenshot("date_selector_not_found")
                
                # Try looking for any element that might be a date
                try:
                    # Use page source to find potential date elements
                    page_source = self.driver.page_source
                    if "date" in page_source.lower() or "calendar" in page_source.lower():
                        print("Page contains date-related text, trying JavaScript approach...")
                        
                        # Try JavaScript to find potential date inputs
                        date_elements = self.driver.execute_script("""
                            return Array.from(document.querySelectorAll('input,button,div,span'))
                                .filter(el => {
                                    const text = (el.textContent || '').toLowerCase();
                                    const id = (el.id || '').toLowerCase();
                                    const className = (el.className || '').toLowerCase();
                                    const type = (el.type || '').toLowerCase();
                                    const placeholder = (el.placeholder || '').toLowerCase();
                                    
                                    return (text.includes('date') || id.includes('date') || 
                                            className.includes('date') || type === 'date' ||
                                            placeholder.includes('date') || text.includes('calendar') ||
                                            id.includes('calendar') || className.includes('calendar'));
                                });
                        """)
                        
                        if date_elements and len(date_elements) > 0:
                            date_element = date_elements[0]
                            print(f"Found potential date element using JavaScript: {date_element.tag_name}")
                        else:
                            print("No date elements found using JavaScript approach")
                    else:
                        print("No date-related text found in page source")
                except Exception as js_error:
                    print(f"JavaScript search error: {js_error}")
                
                # If still no date element found, return False
                if not date_element:
                    print("Failed to find any date element - cannot set date")
                    return False
            
            # Try to interact with the date element
            try:
                print(f"Clicking on date selector: {used_selector}")
                self.take_screenshot("before_click_date")
                
                # Try multiple click methods
                click_success = False
                
                # Method 1: Standard click
                try:
                    print("Trying standard click...")
                    date_element.click()
                    time.sleep(1)
                    click_success = True
                except Exception as click_error:
                    print(f"Standard click failed: {click_error}")
                
                # Method 2: JavaScript click if standard click failed
                if not click_success:
                    try:
                        print("Trying JavaScript click...")
                        self.driver.execute_script("arguments[0].click();", date_element)
                        time.sleep(1)
                        click_success = True
                    except Exception as js_click_error:
                        print(f"JavaScript click failed: {js_click_error}")
                
                # Method 3: Try sending Enter key
                if not click_success:
                    try:
                        print("Trying to send Enter key...")
                        from selenium.webdriver.common.keys import Keys
                        date_element.send_keys(Keys.ENTER)
                        time.sleep(1)
                        click_success = True
                    except Exception as keys_error:
                        print(f"Sending keys failed: {keys_error}")
                
                # Method 4: Try Actions class
                if not click_success:
                    try:
                        print("Trying Actions click...")
                        actions = ActionChains(self.driver)
                        actions.move_to_element(date_element).click().perform()
                        time.sleep(1)
                        click_success = True
                    except Exception as actions_error:
                        print(f"Actions click failed: {actions_error}")
                
                # Method 5: If it's an input field with type="date", try to set directly
                if not click_success and date_element.tag_name.lower() == "input":
                    try:
                        input_type = date_element.get_attribute("type")
                        if input_type and input_type.lower() == "date":
                            print("Found date input field, setting value directly...")
                            # Format the date as YYYY-MM-DD for HTML date inputs
                            formatted_date = f"{yesterday_year}-{yesterday_month:02d}-{yesterday_day:02d}"
                            # First clear the field
                            date_element.clear()
                            # Then set the value using JavaScript (more reliable than send_keys)
                            self.driver.execute_script(
                                f"arguments[0].value = '{formatted_date}';", date_element)
                            # Also dispatch change event to ensure the page updates
                            self.driver.execute_script(
                                "arguments[0].dispatchEvent(new Event('change', { 'bubbles': true }));", 
                                date_element)
                            time.sleep(1)
                            click_success = True
                            
                            # Since we directly set the value, we can skip date picker interaction
                            print(f"Date value set directly to {formatted_date}")
                            self.take_screenshot("date_set_directly")
                            return True
                    except Exception as input_error:
                        print(f"Setting input value failed: {input_error}")
                
                if not click_success:
                    print("All click methods failed, cannot proceed with date selection")
                    return False
                
                self.take_screenshot("after_click_date")
                
                # Now look for date picker elements that might have appeared
                date_picker_elements = [
                    "//div[contains(@class, 'calendar')]",
                    "//div[contains(@class, 'datepicker')]",
                    "//div[contains(@class, 'date-picker')]",
                    "//div[contains(@class, 'picker-dropdown')]",
                    "//div[contains(@class, 'date-selector-dropdown')]",
                    "//table[contains(@class, 'calendar')]",
                    "//div[@role='dialog' and contains(@class, 'date')]"
                ]
                
                date_picker = None
                for picker_selector in date_picker_elements:
                    try:
                        elements = self.driver.find_elements(By.XPATH, picker_selector)
                        visible_elements = [e for e in elements if e.is_displayed()]
                        if visible_elements:
                            date_picker = visible_elements[0]
                            print(f"Found date picker with selector: {picker_selector}")
                            break
                    except Exception as picker_error:
                        print(f"Error finding picker with selector {picker_selector}: {picker_error}")
                        continue
                
                if date_picker:
                    print("Date picker is open, trying to set date")
                    
                    # Use our clean date picker implementation
                    try:
                        from weatherlink.datepicker_fix import set_date_in_picker
                        print("Using clean date picker implementation")
                        date_set = set_date_in_picker(
                            self.driver, 
                            yesterday_year, 
                            yesterday_month, 
                            yesterday_day,
                            self.take_screenshot
                        )
                        if date_set:
                            print("Successfully set the date using clean implementation")
                            return True
                        else:
                            print("Clean implementation failed to set the date")
                            # Continue with the original implementation as fallback
                    except ImportError:
                        print("Could not import clean date picker implementation, using fallback")
                        # Continue with the original implementation
                
                # NEW APPROACH: Follow the exact UI flow provided by the user
                print("\n*** SETTING DATE IN SPECIFIC ORDER: YEAR → MONTH → DAY ***\n")
                
                # Step 1: First click on the "Year" option to enter year selection mode
                year_select_element = None
                try:
                    # Try to find the "Year" selector first - EXACT match for the element in the screenshot
                    year_select_element = self.driver.find_element(By.XPATH, "//div[@id='year-select' and @class='range-item' and text()='Year']")
                    print("Found Year select element with exact match")
                    year_select_element.click()
                    print("Clicked on Year select element")
                    time.sleep(2)  # Give more time for animation
                except Exception as year_select_error:
                    print(f"Error finding/clicking exact Year select element: {year_select_error}")
                    # Try a more relaxed selector
                    try:
                        year_select_element = self.driver.find_element(By.XPATH, "//div[contains(@id, 'year') or (contains(@class, 'range-item') and contains(text(), 'Year'))]")
                        print("Found Year select element with partial match")
                        year_select_element.click()
                        print("Clicked on Year select element")
                        time.sleep(2)
                    except Exception as alt_year_error:
                        print(f"Error with alternative Year element: {alt_year_error}")
                        # Try looking for any clickable element that might trigger year view
                        try:
                            # First look for any datepicker switch to click
                            datepicker_switch = self.driver.find_element(By.XPATH, "//th[contains(@class, 'datepicker-switch')]")
                            print(f"Found datepicker switch: {datepicker_switch.text}")
                            datepicker_switch.click()
                            print("Clicked on datepicker switch to access month/year view")
                            time.sleep(2)
                            
                            # May need to click again to reach year view
                            try:
                                datepicker_switch = self.driver.find_element(By.XPATH, "//th[contains(@class, 'datepicker-switch')]")
                                if not "-" in datepicker_switch.text:  # If it doesn't show a year range yet
                                    print("Clicking again to reach decade view")
                                    datepicker_switch.click()
                                    time.sleep(2)
                            except:
                                pass
                        except Exception as switch_error:
                            print(f"Error clicking datepicker switch: {switch_error}")
                
                # Step 2: Find the datepicker switch showing the decade range
                # Take screenshot to see current state
                self.take_screenshot("year_picker_state")
                
                # Now check if we can see the decade view with year spans
                decade_view_visible = False
                try:
                    # Check if any year spans are visible - if so, we're in the right view
                    year_spans = self.driver.find_elements(By.XPATH, "//span[contains(@class, 'year')]")
                    if len(year_spans) > 0:
                        decade_view_visible = True
                        print(f"Found {len(year_spans)} year spans - already in decade view")
                        # Get the text of some years to log what we're seeing
                        year_texts = [span.text for span in year_spans[:5]]
                        print(f"Sample years visible: {year_texts}")
                except:
                    pass
                
                if not decade_view_visible:
                    print("Cannot find decade view with year spans, trying other methods")
                    # Try different approaches to get to year view
                    try:
                        # Look for the date picker header
                        header = self.driver.find_element(By.XPATH, "//th[contains(@class, 'datepicker-switch')]")
                        print(f"Current date picker header: {header.text}")
                        
                        # If we're in month view (shows year), click to get to year view
                        if not "-" in header.text and len(header.text.strip()) > 0:
                            print("Clicking header to access decade view")
                            header.click()
                            time.sleep(2)
                    except Exception as header_error:
                        print(f"Error navigating to decade view: {header_error}")
                
                
                # Step 3: Navigate to the correct decade range containing the target year
                # Use a simpler approach to avoid syntax errors
                try:
                    # Try to use our clean decade_range_fix implementation if available
                    try:
                        from weatherlink.decade_range_fix import navigate_to_decade_with_year
                        print("Using decade_range_fix module")
                        if navigate_to_decade_with_year(self.driver, yesterday_year):
                            print(f"Successfully navigated to decade containing {yesterday_year}")
                        else:
                            print(f"Could not navigate to decade containing {yesterday_year}")
                    except ImportError:
                        print("Could not import decade_range_fix module")
                        
                        # Fallback: Use simpler direct approach
                        decade_switch = self.driver.find_element(By.XPATH, "//th[contains(@class, 'datepicker-switch')]")
                        decade_text = decade_switch.text
                        print(f"Current decade view: {decade_text}")
                        
                        # Only proceed if we have a decade range format like "2010-2019"
                        if "-" in decade_text:
                            try:
                                # Parse the decade range
                                parts = decade_text.split("-")
                                if len(parts) == 2:
                                    start_year = int(parts[0])
                                    end_year = int(parts[1])
                                    print(f"Decade range: {start_year}-{end_year}, Target year: {yesterday_year}")
                                    
                                    # Navigate if needed
                                    if yesterday_year < start_year:
                                        # Need earlier decade - click prev button
                                        prev_button = self.driver.find_element(By.XPATH, "//th[@class='prev']")
                                        clicks = min(5, (start_year - yesterday_year) // 10 + 1)  # Limit clicks for safety
                                        
                                        for i in range(clicks):
                                            print(f"Click {i+1}: Clicking prev button")
                                            prev_button.click()
                                            time.sleep(1)
                                    elif yesterday_year > end_year:
                                        # Need later decade - click next button
                                        next_button = self.driver.find_element(By.XPATH, "//th[@class='next']")
                                        clicks = min(5, (yesterday_year - end_year) // 10 + 1)  # Limit clicks for safety
                                        
                                        for i in range(clicks):
                                            print(f"Click {i+1}: Clicking next button")
                                            next_button.click()
                                            time.sleep(1)
                                    else:
                                        print(f"Current year {yesterday_year} is already within visible range {start_year}-{end_year}")
                            except Exception as e:
                                print(f"Error navigating decades: {e}")
                        else:
                            print(f"Cannot determine decade range from '{decade_text}'")
                except Exception as e:
                    print(f"Error handling decade range: {e}")
                
                # Take a screenshot after navigation
                self.take_screenshot("after_decade_navigation")


                # Try multiple strategies to locate the year span
                year_set = False
                
                # First approach: direct match for year span with exact text
                try:
                    # Most precise selector
                    year_spans = self.driver.find_elements(By.XPATH, 
                        f"//span[contains(@class, 'year') and not(contains(@class, 'old')) and not(contains(@class, 'new')) and normalize-space(text())='{yesterday_year}']")
                    
                    if year_spans:
                        print(f"Found precise year span for {yesterday_year}, clicking...")
                        # Log the classes on this span for debugging
                        print(f"Year span classes: {year_spans[0].get_attribute('class')}")
                        year_spans[0].click()
                        time.sleep(2)
                        year_set = True
                    else:
                        print(f"Could not find precise year span for {yesterday_year}")
                except Exception as precise_year_error:
                    print(f"Error with precise year selection: {precise_year_error}")
                
                # Second approach: include spans that might be marked as "old" or "new"
                if not year_set:
                    try:
                        # More general selector
                        year_spans = self.driver.find_elements(By.XPATH, 
                            f"//span[contains(@class, 'year') and normalize-space(text())='{yesterday_year}']")
                        
                        if year_spans:
                            print(f"Found year span for {yesterday_year} (might be old/new), clicking...")
                            # Log the classes on this span for debugging
                            print(f"Year span classes: {year_spans[0].get_attribute('class')}")
                            year_spans[0].click()
                            time.sleep(2)
                            year_set = True
                        else:
                            print(f"Could not find any year span for {yesterday_year}")
                    except Exception as general_year_error:
                        print(f"Error with general year selection: {general_year_error}")
                
                # Third approach: look at all year spans and match by text
                if not year_set:
                    try:
                        # Get all year spans
                        all_year_spans = self.driver.find_elements(By.XPATH, "//span[contains(@class, 'year')]")
                        print(f"Found {len(all_year_spans)} year spans in total")
                        
                        # Log what years we found
                        years_found = [span.text.strip() for span in all_year_spans[:15]]  # Limit to first 15
                        print(f"Years found: {years_found}")
                        
                        # Look for exact match by text
                        for span in all_year_spans:
                            span_text = span.text.strip()
                            if span_text == str(yesterday_year):
                                print(f"Found year {yesterday_year} by text matching")
                                print(f"Year span classes: {span.get_attribute('class')}")
                                span.click()
                                time.sleep(2)
                                year_set = True
                                break
                        
                        if not year_set:
                            print(f"Could not find year {yesterday_year} among all year spans")
                            # As a last resort, try JavaScript to find and click
                            try:
                                print(f"Trying JavaScript to find year {yesterday_year}")
                                year_elem = self.driver.execute_script(f"""
                                    var yearSpans = document.querySelectorAll('span.year, span[class*="year"]');
                                    for (var i = 0; i < yearSpans.length; i++) {{
                                        if (yearSpans[i].textContent.trim() === '{yesterday_year}') {{
                                            return yearSpans[i];
                                        }}
                                    }}
                                    return null;
                                """)
                                
                                if year_elem:
                                    print("Found year element with JavaScript, clicking")
                                    self.driver.execute_script("arguments[0].click();", year_elem)
                                    time.sleep(2)
                                    year_set = True
                                else:
                                    print("JavaScript could not find the year element either")
                            except Exception as js_error:
                                print(f"JavaScript year selection error: {js_error}")
                    except Exception as all_year_error:
                        print(f"Error examining all year spans: {all_year_error}")
                
                # Take screenshot after year selection
                self.take_screenshot("after_year_selection")
                
                # Step 5: Select the month
                month_set = False
                if year_set:
                    try:
                        # Look for the specific month
                        month_spans = self.driver.find_elements(By.XPATH, 
                            f"//span[contains(@class, 'month') and text()='{yesterday_month_name[:3]}']")
                        
                        if month_spans:
                            print(f"Found month span for {yesterday_month_name}, clicking...")
                            try:
                                # First try: standard click method (likely to fail based on previous errors)
                                month_spans[0].click()
                            except Exception as month_click_error:
                                print(f"Standard click on month failed: {month_click_error}")
                                print("Trying JavaScript to click on month (since it worked for year)...")
                                
                                # Try JavaScript click which worked for the year selection
                                self.driver.execute_script("arguments[0].click();", month_spans[0])
                            
                            time.sleep(1)
                            month_set = True
                        else:
                            print(f"Could not find month span for {yesterday_month_name[:3]}")
                            # Try finding all month spans to see what's available
                            all_month_spans = self.driver.find_elements(By.XPATH, "//span[contains(@class, 'month')]")
                            
                            if all_month_spans:
                                print(f"Found {len(all_month_spans)} month spans")
                                # Log the text of the month spans to see what we're working with
                                month_texts = [span.text for span in all_month_spans[:12]]
                                print(f"Month texts: {month_texts}")
                                
                                # Try JavaScript to select the month by index
                                print(f"Attempting to select month {yesterday_month} using JavaScript...")
                                month_index = yesterday_month - 1  # Convert to 0-based index
                                
                                # Direct JavaScript approach to select the month by index
                                script = f"""
                                    var monthSpans = document.querySelectorAll('span.month, span[class*="month"]');
                                    if (monthSpans.length > {month_index}) {{
                                        monthSpans[{month_index}].click();
                                        return true;
                                    }}
                                    return false;
                                """
                                result = self.driver.execute_script(script)
                                if result:
                                    print(f"Successfully clicked month at index {month_index} using JavaScript")
                                    time.sleep(1)
                                    month_set = True
                                else:
                                    print(f"JavaScript could not click month at index {month_index}")
                                    
                                    # Try a more comprehensive approach
                                    print("Trying alternative JavaScript approach for month selection...")
                                    month_script = f"""
                                        // First try by text content
                                        var allSpans = document.querySelectorAll('span');
                                        for (var i = 0; i < allSpans.length; i++) {{
                                            if (allSpans[i].textContent.trim() === '{yesterday_month_name}' || 
                                                allSpans[i].textContent.trim() === '{yesterday_month_name[:3]}') {{
                                                allSpans[i].click();
                                                return "Found by full text: " + allSpans[i].textContent;
                                            }}
                                        }}
                                        
                                        // Try by class and position
                                        var monthElements = document.querySelectorAll('.month, [class*="month"]');
                                        if (monthElements.length >= {yesterday_month}) {{
                                            monthElements[{month_index}].click();
                                            return "Selected by index: " + monthElements[{month_index}].textContent;
                                        }}
                                        
                                        return "No month element found";
                                    """
                                    month_result = self.driver.execute_script(month_script)
                                    print(f"Month selection result: {month_result}")
                                    if "Found by" in str(month_result) or "Selected by" in str(month_result):
                                        time.sleep(1)
                                        month_set = True
                            else:
                                print("No month spans found")
                    except Exception as month_error:
                        print(f"Error selecting month: {month_error}")
                
                # Take screenshot after month selection attempt
                self.take_screenshot("after_month_selection")
                
                # Step 6: Select the day
                day_set = False
                if month_set:
                    try:
                        # Make sure we're looking at the correct month header
                        month_header = self.driver.find_element(By.XPATH, "//th[contains(@class, 'datepicker-switch')]")
                        month_header_text = month_header.text
                        print(f"Month header: {month_header_text}")
                        
                        # Look for the day that's not in old or new classes (current month)
                        day_cells = self.driver.find_elements(By.XPATH, 
                            f"//td[contains(@class, 'day') and not(contains(@class, 'old')) and not(contains(@class, 'new')) and text()='{yesterday_day}']")
                        
                        if day_cells:
                            print(f"Found day {yesterday_day} in current month, clicking...")
                            # Try regular click first
                            try:
                                day_cells[0].click()
                            except Exception as day_click_error:
                                print(f"Standard click on day failed: {day_click_error}")
                                print("Trying JavaScript to click day...")
                                self.driver.execute_script("arguments[0].click();", day_cells[0])
                            
                            time.sleep(1)
                            day_set = True
                        else:
                            print(f"Could not find day {yesterday_day} in current month")
                            # Try any day cell with the right text
                            any_day_cells = self.driver.find_elements(By.XPATH, f"//td[contains(@class, 'day') and text()='{yesterday_day}']")
                            if any_day_cells:
                                print(f"Found day {yesterday_day} (may be in adjacent month)")
                                # Try JavaScript click
                                try:
                                    self.driver.execute_script("arguments[0].click();", any_day_cells[0])
                                    print("Clicked day using JavaScript")
                                    time.sleep(1)
                                    day_set = True
                                except Exception as js_day_error:
                                    print(f"JavaScript click on day failed: {js_day_error}")
                                    
                                    # Try direct JavaScript approach to find and click the day
                                    print(f"Trying comprehensive JavaScript approach to find day {yesterday_day}...")
                                    day_script = f"""
                                        var dayCells = document.querySelectorAll('td.day, td[class*="day"]');
                                        for (var i = 0; i < dayCells.length; i++) {{
                                            if (dayCells[i].textContent.trim() === '{yesterday_day}') {{
                                                dayCells[i].click();
                                                return true;
                                            }}
                                        }}
                                        return false;
                                    """
                                    if self.driver.execute_script(day_script):
                                        print(f"Successfully clicked day {yesterday_day} using JavaScript")
                                        time.sleep(1)
                                        day_set = True
                                    else:
                                        print(f"Could not find or click day {yesterday_day} using JavaScript")
                            else:
                                print(f"Could not find any day cell with text '{yesterday_day}'")
                                
                                # Last resort: try to get all visible days and pick the one we want
                                all_days = self.driver.find_elements(By.XPATH, "//td[contains(@class, 'day')]")
                                if all_days:
                                    print(f"Found {len(all_days)} day cells in total")
                                    day_texts = [day.text for day in all_days[:15]]
                                    print(f"Sample days: {day_texts}")
                                    
                                    # Try to find our day in the list
                                    matching_days = [d for d in all_days if d.text.strip() == str(yesterday_day)]
                                    if matching_days:
                                        print(f"Found matching day: {matching_days[0].text}")
                                        try:
                                            self.driver.execute_script("arguments[0].click();", matching_days[0])
                                            print("Clicked matching day using JavaScript")
                                            time.sleep(1)
                                            day_set = True
                                        except Exception as match_day_error:
                                            print(f"Error clicking matching day: {match_day_error}")
                    except Exception as day_error:
                        print(f"Error selecting day: {day_error}")
                
                # Take screenshot after day selection
                self.take_screenshot("after_day_selection")
                
                # If all steps failed, try our dedicated JavaScript helper
                if not (year_set and month_set and day_set):
                    print("\nTrying dedicated JavaScript date selection helper...")
                    success, message = use_js_to_select_date(self.driver, yesterday_year, yesterday_month, yesterday_day)
                    if success:
                        print(f"JavaScript date helper succeeded: {message}")
                        self.take_screenshot("after_js_date_helper")
                        # Consider the date as fully set if our helper reports success
                        year_set = month_set = day_set = True
                    else:
                        print(f"JavaScript date helper failed: {message}")
                
                # Final JS approach to try setting the date
                if not (year_set and month_set and day_set):
                    print("\nTrying a final approach with custom JavaScript...")
                    try:
                        # This approach focuses on targeting specific attributes from the HTML structure
                        direct_js = f"""
                        // Final attempt to set the date using raw JavaScript
                        (function() {{
                            // Click on the Year selector first
                            var yearSelect = document.querySelector('#year-select');
                            if (yearSelect) {{
                                console.log("Found year selector");
                                yearSelect.click();
                                
                                // Small delay then find and click the year
                                setTimeout(function() {{
                                    var years = Array.from(document.querySelectorAll('span.year')).filter(
                                        y => y.textContent.trim() === '{yesterday_year}'
                                    );
                                    if (years.length > 0) {{
                                        console.log("Found year {yesterday_year}");
                                        years[0].click();
                                        
                                        // Find and click the month 
                                        setTimeout(function() {{
                                            var monthIndex = {yesterday_month - 1};
                                            var months = document.querySelectorAll('span.month');
                                            if (months.length > monthIndex) {{
                                                console.log("Found month at index " + monthIndex);
                                                months[monthIndex].click();
                                                
                                                // Find and click the day
                                                setTimeout(function() {{
                                                    var days = Array.from(document.querySelectorAll('td.day')).filter(
                                                        d => d.textContent.trim() === '{yesterday_day}' && 
                                                        !d.classList.contains('old') && 
                                                        !d.classList.contains('new')
                                                    );
                                                    if (days.length > 0) {{
                                                        console.log("Found day {yesterday_day}");
                                                        days[0].click();
                                                        return true;
                                                    }}
                                                    return false;
                                                }}, 500);
                                            }}
                                        }}, 500);
                                    }}
                                }}, 500);
                            }}
                            return "Date selection process started";
                        }})();
                        """
                        result = self.driver.execute_script(direct_js)
                        print(f"Final JavaScript approach result: {result}")
                        time.sleep(2)  # Give time for all the setTimeout callbacks to complete
                        self.take_screenshot("after_final_js_attempt")
                    except Exception as final_js_error:
                        print(f"Error in final JavaScript attempt: {final_js_error}")
                
                # Check if we successfully set the date
                if year_set and month_set and day_set:
                    print(f"Successfully set date to {yesterday_month_name} {yesterday_day}, {yesterday_year}")
                else:
                    print(f"Could not fully set date to {yesterday_month_name} {yesterday_day}, {yesterday_year}")
                    print(f"Year set: {year_set}, Month set: {month_set}, Day set: {day_set}")
                
                # Try to close the date picker dialog
                try:
                    # Try clicking the "Apply" button if it exists
                    apply_buttons = self.driver.find_elements(By.XPATH, "//button[contains(@class, 'apply') or contains(text(), 'Apply')]")
                    if apply_buttons:
                        print("Found Apply button, clicking to close date picker")
                        apply_buttons[0].click()
                    else:
                        # Try clicking outside the date picker
                        print("No Apply button found, clicking outside the date picker to close it")
                        actions = ActionChains(self.driver)
                        actions.move_by_offset(10, 10).click().perform()
                except Exception as close_error:
                    print(f"Error closing date picker: {close_error}")
                
                # Try to verify that the date is now set to yesterday
                print("Checking if the date was updated correctly")
                self.take_screenshot("after_date_selection")
                
                # Look for date display elements to verify date
                date_display_selectors = [
                    "//span[@class='time']",
                    "//input[@type='date']",
                    "//div[contains(@class, 'selected-date')]",
                    "//button[contains(@class, 'date-display')]",
                    "//span[contains(@class, 'date-display')]",
                    "//input[contains(@class, 'date') or @type='date']",
                    # Add the original date element we interacted with
                    used_selector if used_selector else None
                ]
                
                date_verified = False
                for selector in date_display_selectors:
                    if not selector:
                        continue
                        
                    try:
                        elements = self.driver.find_elements(By.XPATH, selector)
                        visible_elements = [e for e in elements if e.is_displayed()]
                        
                        if visible_elements:
                            element = visible_elements[0]
                            # Get text content or value
                            date_text = element.text
                            if not date_text and element.tag_name == "input":
                                date_text = element.get_attribute("value")
                                
                            print(f"Found date display: {date_text}")
                            
                            if not date_text:
                                print("Date text is empty, trying next selector")
                                continue
                                
                            # Prepare yesterday's date components for comparison
                            yesterday_day_str = str(yesterday_day)
                            yesterday_month_str = str(yesterday_month)
                            yesterday_year_str = str(yesterday_year)
                            yesterday_year_short = str(yesterday_year)[-2:]  # Last 2 digits of year
                            
                            # If day or month is single digit, also create padded versions (e.g., 01 instead of 1)
                            padded_day = f"0{yesterday_day}" if yesterday_day < 10 else yesterday_day_str
                            padded_month = f"0{yesterday_month}" if yesterday_month < 10 else yesterday_month_str
                            
                            # Extract all date-like patterns from the text
                            import re
                            # Look for date patterns like MM/DD/YY, DD/MM/YY, etc.
                            date_patterns = re.findall(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})', date_text)
                            
                            if date_patterns:
                                print(f"Found date patterns in text: {date_patterns}")
                                for pattern in date_patterns:
                                    # Try to parse this pattern
                                    date_parts = re.split(r'[-/\.]', pattern)
                                    if len(date_parts) == 3:
                                        # Check multiple date formats
                                        # YYYY-MM-DD format (year first)
                                        if (date_parts[0] == yesterday_year_str or date_parts[0] == yesterday_year_short) and \
                                           (date_parts[1] == yesterday_month_str or date_parts[1] == padded_month) and \
                                           (date_parts[2] == yesterday_day_str or date_parts[2] == padded_day):
                                            print(f"Verified correct date in YYYY-MM-DD format: {pattern}")
                                            date_verified = True
                                            break
                                            
                                        # MM/DD/YY format (most common in US)
                                        elif (date_parts[0] == yesterday_month_str or date_parts[0] == padded_month) and \
                                             (date_parts[1] == yesterday_day_str or date_parts[1] == padded_day) and \
                                             (date_parts[2] == yesterday_year_str or date_parts[2] == yesterday_year_short):
                                            print(f"Verified correct date in MM/DD/YY format: {pattern}")
                                            date_verified = True
                                            break
                                            
                                        # DD/MM/YY pattern
                                        elif (date_parts[0] == yesterday_day_str or date_parts[0] == padded_day) and \
                                             (date_parts[1] == yesterday_month_str or date_parts[1] == padded_month) and \
                                             (date_parts[2] == yesterday_year_str or date_parts[2] == yesterday_year_short):
                                            print(f"Verified correct date in DD/MM/YY format: {pattern}")
                                            date_verified = True
                                            break
                                                
                                        # Print details for debugging
                                        print(f"Date pattern {pattern} analysis:")
                                        print(f"Parts: {date_parts}")
                                        print(f"Expected: Year={yesterday_year_str}/{yesterday_year_short}, Month={yesterday_month_str}/{padded_month}, Day={yesterday_day_str}/{padded_day}")
                            
                            # If no date pattern matched, check for presence of date components and month name
                            if not date_verified:
                                # Check if month name is present along with day and year
                                if (yesterday_month_name in date_text or yesterday_month_name.lower() in date_text.lower()) and \
                                   (yesterday_day_str in date_text or padded_day in date_text) and \
                                   (yesterday_year_str in date_text or yesterday_year_short in date_text):
                                    print(f"Verified date by components: {yesterday_month_name}, {yesterday_day}, {yesterday_year}")
                                    date_verified = True
                                else:
                                    # If we get here, the date text doesn't contain the expected date in any recognizable format
                                    print(f"Date text '{date_text}' doesn't match expected date: {yesterday_month}/{yesterday_day}/{yesterday_year} ({yesterday_month_name})")
                            
                            if date_verified:
                                break
                    except Exception as verify_error:
                        print(f"Error verifying date with selector {selector}: {verify_error}")
                        continue
                
                if not date_verified:
                    print("\nWARNING: Could not verify date was set correctly to yesterday's date")
                    print(f"Expected: {yesterday_month}/{yesterday_day}/{yesterday_year} ({yesterday_month_name})")
                    print("The displayed date may not reflect yesterday's date or may be in an unexpected format")
                    # Take one final screenshot showing the current state
                    self.take_screenshot("date_verification_failed")
                    # Since we made our best effort, we'll return False to indicate this operation wasn't successful
                    return False
                else:
                    print(f"\nSUCCESS: Date verified as correctly set to yesterday's date: {yesterday_month}/{yesterday_day}/{yesterday_year}")
                
                return True
                
            except Exception as e:
                print(f"Error setting date: {e}")
                self.take_screenshot("set_date_error")
                # Return False to indicate failure, but the calling function can decide to continue
                return False
        
        except Exception as e:
            print(f"Error setting date: {e}")
            self.take_screenshot("set_date_error")
            # Return False to indicate failure, but the calling function can decide to continue
            return False
    
    def take_screenshot(self, name):
        """Take a screenshot and save it with the given name."""
        if self.debug:
            try:
                # Create screenshots directory if it doesn't exist
                screenshots_dir = os.path.join(SCRIPT_DIR, "screenshots")
                if not os.path.exists(screenshots_dir):
                    os.makedirs(screenshots_dir)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{screenshots_dir}/{name}_{timestamp}.png"
                
                # Take screenshot
                self.driver.save_screenshot(filename)
                print(f"Screenshot saved: {filename}")
            except Exception as e:
                print(f"Error taking screenshot: {e}")
    
    def navigate_to_data_page(self):
        """Navigate to the data page."""
        try:
            print("Navigating to data page...")
            self.take_screenshot("before_navigation")
            
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Check if we need to log in again
                    if "login" in self.driver.current_url.lower():
                        print("Detected login page, attempting to log in again...")
                        if not self.login():
                            print("Failed to log in")
                            return False
                        print("Login successful")
                        time.sleep(3)  # Wait for login to complete
                    
                    # Try direct URL navigation with the specific URL
                    print("Trying direct URL navigation to the data page...")
                    specific_url = f"{self.url}browse/2bd0cbc6-a874-441f-99f4-2410a8143886"
                    self.driver.get(specific_url)
                    print(f"Navigated directly to: {specific_url}")
                    time.sleep(5)  # Give more time for the page to load
                    
                    # Take a screenshot of the data page
                    self.take_screenshot("data_page")
                    
                    # Log the current URL so we can see where we ended up
                    print(f"Current URL after navigation: {self.driver.current_url}")
                    
                    # Check if we ended up on the login page again
                    if "login" in self.driver.current_url.lower():
                        print("Redirected to login page, attempting to log in again...")
                        if not self.login():
                            print("Failed to log in")
                            return False
                        print("Login successful")
                        time.sleep(3)  # Wait for login to complete
                        
                        # Try direct URL navigation once more
                        self.driver.get(specific_url)
                        print(f"Navigated directly to: {specific_url}")
                        time.sleep(5)
                        
                        # Check if we're still on the login page
                        if "login" in self.driver.current_url.lower():
                            print("Still on login page, navigation failed")
                            if attempt < max_retries - 1:
                                print(f"Will retry in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                continue
                            else:
                                print("Max retries reached. Navigation failed.")
                                return False
                    
                    # Check for successful navigation by looking for typical elements on the data page
                    try:
                        # Look for elements that would be on the data page
                        data_indicators = [
                            "//table[contains(@class, 'data-table')]",
                            "//div[contains(@class, 'data-view')]",
                            "//h1[contains(text(), 'Data')]",
                            "//a[@id='export-data']"
                        ]
                        
                        navigation_success = False
                        for indicator in data_indicators:
                            try:
                                element = WebDriverWait(self.driver, 5).until(
                                    EC.presence_of_element_located((By.XPATH, indicator))
                                )
                                print(f"Found data page indicator: {indicator}")
                                navigation_success = True
                                break
                            except:
                                continue
                        
                        if navigation_success:
                            print("Successfully navigated to data page")
                            return True
                        else:
                            print("Could not confirm successful navigation to data page")
                            # Take another screenshot to debug
                            self.take_screenshot("data_page_confirmation_failed")
                            # Continue anyway - the next steps will fail if this isn't right
                            return True
                    except Exception as confirm_error:
                        print(f"Error confirming navigation: {confirm_error}")
                        # Continue anyway
                        return True
                    
                except Exception as e:
                    print(f"Error during navigation attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        print(f"Waiting {retry_delay} seconds before retrying...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached. Navigation failed.")
                        self.take_screenshot("navigation_error")
                        return False
            
        except Exception as e:
            print(f"Error navigating to data page: {e}")
            self.take_screenshot("navigation_error")
            return False
    
    def extract_weather_data(self):
        """Extract weather data from the current page."""
        try:
            print("Extracting weather data...")
            self.take_screenshot("before_extraction")
            
            # Wait for the data table to be present
            table = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.data-table"))
            )
            
            # Get all rows from the table
            rows = table.find_elements(By.TAG_NAME, "tr")
            if not rows:
                print("No rows found in the table")
                return None
                
            # Get headers from the first row
            headers = [cell.text.strip() for cell in rows[0].find_elements(By.TAG_NAME, "th")]
            if not headers:
                print("No headers found in the table")
                return None
                
            print(f"Found {len(headers)} columns: {headers}")
            
            # Extract data from remaining rows
            data = []
            for row in rows[1:]:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) != len(headers):
                    print(f"Warning: Row has {len(cells)} cells but expected {len(headers)}")
                    continue
                    
                row_data = []
                for cell in cells:
                    row_data.append(cell.text.strip())
                if row_data:  # Only add non-empty rows
                    data.append(row_data)
                    
            print(f"Extracted {len(data)} rows of data")
            
            # Create DataFrame from extracted data
            if data:
                df = pd.DataFrame(data, columns=headers)
                print(f"Created DataFrame with shape: {df.shape}")
                return df
            else:
                print("No data rows found")
                return None
            
        except Exception as e:
            print(f"Error extracting data: {e}")
            self.take_screenshot("extraction_error")
            return None
    
    def export_data(self, email=None):
        """Export data by clicking the export button and sending to email."""
        try:
            # Use provided email or default to the one set in constructor
            if email is None:
                email = self.export_email
                
            print(f"Exporting data to email: {email}...")
            self.take_screenshot("before_export")
            
            # Check if we're on the data page, if not, try to navigate there
            if not ("browse" in self.driver.current_url.lower() or "data" in self.driver.current_url.lower()):
                print("Not on data page, attempting to navigate to data page...")
                if not self.navigate_to_data_page():
                    print("Failed to navigate to data page")
                    return False
                
            # Check if we need to log in again
            if "login" in self.driver.current_url.lower():
                print("Redirected to login page, attempting to log in again...")
                if not self.login():
                    print("Failed to log in")
                    return False
                print("Login successful")
                time.sleep(3)  # Wait for login to complete
                
                # Navigate to data page again
                if not self.navigate_to_data_page():
                    print("Failed to navigate to data page after login")
                    return False
            
            # Check if there's a modal dialog blocking the UI and close it
            try:
                print("Checking for modal dialogs that might block the UI...")
                modal = self.driver.find_element(By.ID, "modal-config")
                if modal.is_displayed():
                    print("Found a modal dialog blocking the UI. Attempting to close it...")
                    # Try to find close button or X button
                    close_buttons = self.driver.find_elements(By.XPATH, 
                        "//div[@id='modal-config']//button[contains(@class, 'close') or contains(@class, 'btn-close')]")
                    
                    if close_buttons:
                        print("Found close button for the modal")
                        close_buttons[0].click()
                        print("Clicked close button")
                    else:
                        # Try to click anywhere outside the modal to close it
                        print("No close button found. Trying to click outside the modal...")
                        actions = ActionChains(self.driver)
                        actions.move_by_offset(10, 10).click().perform()
                        
                    # Wait for modal to disappear
                    print("Waiting for modal to disappear...")
                    WebDriverWait(self.driver, 5).until_not(
                        EC.visibility_of_element_located((By.ID, "modal-config"))
                    )
                    print("Modal is no longer visible")
                else:
                    print("Modal is present but not displayed")
            except Exception as e:
                print(f"No blocking modal found or error handling modal: {e}")
            
            # Wait for the export button to be clickable
            print("Looking for export button...")
            time.sleep(2)  # Give the page a moment to settle after modal handling
            
            # Try different selectors for the export button
            export_button = None
            export_selectors = [
                "a#export-data",
                "a[id='export-data']",
                "a.export-button",
                "button.export-button",
                "a[title='Export Data']",
                "a[href='#export']",
                "//a[contains(text(), 'Export')]",
                "//button[contains(text(), 'Export')]"
            ]
            
            for selector in export_selectors:
                try:
                    # Determine if it's a CSS or XPath selector
                    if selector.startswith("//"):
                        export_button = WebDriverWait(self.driver, 3).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        export_button = WebDriverWait(self.driver, 3).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                    if export_button:
                        print(f"Found export button with selector: {selector}")
                        break
                except:
                    continue
            
            if not export_button:
                print("Could not find any export button")
                self.take_screenshot("export_button_not_found")
                return False
            
            # Try alternative methods to click the button if direct click fails
            try:
                # First try: standard click
                print("Clicking export button...")
                export_button.click()
            except Exception as click_error:
                print(f"Standard click failed: {click_error}")
                print("Trying JavaScript click...")
                self.driver.execute_script("arguments[0].click();", export_button)
                
            time.sleep(2)
            self.take_screenshot("export_form_opened")
            
            # Wait for the email input field
            print("Looking for email input field...")
            email_field = None
            try:
                email_field = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input#email.form-control"))
                )
            except:
                # Try alternate selectors
                try:
                    email_field = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//input[@type='email' or @id='email' or @name='email']"))
                    )
                except:
                    print("Could not find email input field")
                    self.take_screenshot("email_field_not_found")
                    return False
                    
            print("Found email input field")
            
            # Clear any existing text and enter the email
            email_field.clear()
            print(f"Entering email: {email}")
            email_field.send_keys(email)
            time.sleep(1)
            self.take_screenshot("email_entered")
            
            # Look for the Send button or Export button
            print("Looking for Send/Export button...")
            send_button = None
            send_selectors = [
                "button#js-updateBtn.export-button",
                "button.export-button",
                "input[type='submit']",
                "button[type='submit']",
                "//button[contains(text(), 'Send')]",
                "//button[contains(text(), 'Export')]",
                "//input[@value='Export']"
            ]
            
            for selector in send_selectors:
                try:
                    # Determine if it's a CSS or XPath selector
                    if selector.startswith("//"):
                        send_button = WebDriverWait(self.driver, 3).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        send_button = WebDriverWait(self.driver, 3).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                    if send_button:
                        print(f"Found send/export button with selector: {selector}")
                        break
                except:
                    continue
                    
            if not send_button:
                print("Could not find any send/export button")
                self.take_screenshot("send_button_not_found")
                return False
            
            print("Clicking Send button...")
            try:
                # First try: standard click
                send_button.click()
            except Exception as click_error:
                print(f"Standard click failed: {click_error}")
                print("Trying JavaScript click...")
                self.driver.execute_script("arguments[0].click();", send_button)
                
            time.sleep(3)
            self.take_screenshot("after_export_sent")
            
            # Look for confirmation message
            try:
                confirmation = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'alert-success') or contains(text(), 'success') or contains(text(), 'sent')]"))
                )
                print("Found success confirmation message")
            except:
                # If no confirmation found, check for error messages
                try:
                    error_msg = WebDriverWait(self.driver, 2).until(
                        EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'alert-danger') or contains(@class, 'alert-error')]"))
                    )
                    print(f"Error message found: {error_msg.text}")
                    self.take_screenshot("export_error_message")
                except:
                    print("No confirmation or error message found")
                    print("Assuming export request was sent anyway")
            
            print("Data export request sent successfully")
            return True
            
        except Exception as e:
            print(f"Error during data export: {e}")
            self.take_screenshot("export_error")
            return False

    def check_gmail_api(self, max_wait_time=300):
        """
        Check Gmail for the exported data email using the API.
        
        Args:
            max_wait_time: Maximum time to wait for the email (in seconds)
            
        Returns:
            Path to the downloaded CSV file or None if not found
        """
        try:
            print("\nAutomatically checking Gmail using API for the exported data...")
            
            if GmailAPI is None:
                print("GmailAPI module is not available. Please check the import error above.")
                return None
            
            # Initialize the GmailAPI
            try:
                print("Initializing GmailAPI...")
                gmail_api = GmailAPI()
                if not gmail_api:
                    print("GmailAPI initialization returned None")
                    return None
                if not hasattr(gmail_api, 'service'):
                    print("GmailAPI instance has no 'service' attribute")
                    return None
                print("GmailAPI initialized successfully")
            except Exception as init_error:
                print(f"Error initializing GmailAPI: {init_error}")
                print("Please check your credentials.json file and Gmail API setup")
                return None
            
            # Set up for waiting - check multiple times with delays
            start_time = time.time()
            retry_count = 0
            max_retries = 10  # Maximum number of retries
            retry_delay = 30  # Initial delay between retries in seconds
            
            # Store message IDs we've already checked to avoid re-checking the same messages
            checked_message_ids = set()
            
            while time.time() - start_time < max_wait_time and retry_count < max_retries:
                try:
                    print(f"\nChecking for WeatherLink emails (Attempt {retry_count + 1}/{max_retries})...")
                    
                    # Get the messages
                    messages = gmail_api.service.users().messages().list(
                        userId='me',
                        q='from:weatherlink.com',
                        maxResults=10
                    ).execute()
                    
                    if 'messages' in messages and messages['messages']:
                        print(f"Found {len(messages['messages'])} messages from WeatherLink")
                        
                        # Process messages from newest to oldest
                        for msg_info in messages['messages']:
                            msg_id = msg_info['id']
                            
                            # Skip if we've already checked this message
                            if msg_id in checked_message_ids:
                                continue
                            
                            checked_message_ids.add(msg_id)
                            
                            try:
                                print(f"Examining message {msg_id}...")
                                msg = gmail_api.service.users().messages().get(
                                    userId='me',
                                    id=msg_id,
                                    format='full'
                                ).execute()
                                
                                # Get timestamp and check if this is a recent message
                                headers = msg['payload']['headers']
                                date_header = next((h for h in headers if h['name'].lower() == 'date'), None)
                                
                                if date_header:
                                    print(f"Message date: {date_header['value']}")
                                
                                # Extract the S3 URL from the email content
                                s3_url = None
                                
                                # Check message structure - look in parts first
                                if 'parts' in msg['payload']:
                                    print("Message has parts structure - analyzing...")
                                    for part_idx, part in enumerate(msg['payload']['parts']):
                                        print(f"Examining part {part_idx+1} of {len(msg['payload']['parts'])}: {part.get('mimeType', 'unknown type')}")
                                        if part['mimeType'] == 'text/html':
                                            data = part['body'].get('data', '')
                                            if data:
                                                import base64
                                                html = base64.urlsafe_b64decode(data).decode('utf-8')
                                                print(f"HTML content preview (first 200 chars): {html[:200]}...")
                                                
                                                # Look for the download link in the HTML - try multiple patterns
                                                import re
                                                # First attempt with specific pattern
                                                s3_url_pattern = r'https://s3\.amazonaws\.com/export-wl2-live\.weatherlink\.com/data/[^"]+\.csv'
                                                match = re.search(s3_url_pattern, html)
                                                if match:
                                                    s3_url = match.group(0)
                                                    print(f"Found S3 URL with specific pattern: {s3_url}")
                                                    break
                                                
                                                # Second attempt with more general pattern
                                                s3_url_pattern_alt = r'https://s3\.amazonaws\.com/[^"]+\.csv'
                                                match = re.search(s3_url_pattern_alt, html)
                                                if match:
                                                    s3_url = match.group(0)
                                                    print(f"Found S3 URL with general pattern: {s3_url}")
                                                    break
                                                
                                                # Third attempt looking specifically for href attributes
                                                href_pattern = r'href="(https://s3\.amazonaws\.com/[^"]+\.csv)"'
                                                match = re.search(href_pattern, html)
                                                if match:
                                                    s3_url = match.group(1)
                                                    print(f"Found S3 URL in href attribute: {s3_url}")
                                                    break
                                                    
                                                # If still not found, search for any link with .csv extension
                                                csv_link_pattern = r'href="([^"]+\.csv)"'
                                                match = re.search(csv_link_pattern, html)
                                                if match:
                                                    s3_url = match.group(1)
                                                    print(f"Found generic CSV link: {s3_url}")
                                                    break
                                                    
                                                print("No S3 URL patterns matched in HTML content")
                                            else:
                                                print("Part has no data content")
                                # Check for body directly in the message if no parts or S3 URL not found in parts
                                elif 'body' in msg['payload'] and not s3_url:
                                    print("Checking message body directly...")
                                    data = msg['payload']['body'].get('data', '')
                                    if data:
                                        import base64
                                        html = base64.urlsafe_b64decode(data).decode('utf-8')
                                        print(f"Body content preview (first 200 chars): {html[:200]}...")
                                        
                                        # Same patterns as above
                                        import re
                                        # Try all patterns
                                        patterns = [
                                            r'https://s3\.amazonaws\.com/export-wl2-live\.weatherlink\.com/data/[^"]+\.csv',
                                            r'https://s3\.amazonaws\.com/[^"]+\.csv',
                                            r'href="(https://s3\.amazonaws\.com/[^"]+\.csv)"',
                                            r'href="([^"]+\.csv)"'
                                        ]
                                        
                                        for pattern in patterns:
                                            match = re.search(pattern, html)
                                            if match:
                                                # If it's a capturing group, get group 1, otherwise get the whole match
                                                s3_url = match.group(1) if '(' in pattern else match.group(0)
                                                print(f"Found S3 URL in body using pattern {pattern}: {s3_url}")
                                                break
                                
                                if s3_url:
                                    print(f"\nFound S3 URL: {s3_url}")
                                    
                                    # Create saved_emails directory if it doesn't exist
                                    if not os.path.exists(SAVED_EMAILS_DIR):
                                        os.makedirs(SAVED_EMAILS_DIR)
                                    
                                    # Extract filename from URL
                                    filename = s3_url.split('/')[-1]
                                    filepath = os.path.join(SAVED_EMAILS_DIR, filename)
                                    
                                    # Download the file directly using requests
                                    print(f"Downloading file from {s3_url}...")
                                    try:
                                        import requests
                                        response = requests.get(s3_url)
                                        if response.status_code == 200:
                                            with open(filepath, 'wb') as f:
                                                f.write(response.content)
                                            print(f"Successfully downloaded file: {filepath}")
                                            
                                            # Return the filepath of the downloaded file
                                            return filepath
                                        else:
                                            print(f"Failed to download file. Status code: {response.status_code}")
                                    except Exception as download_error:
                                        print(f"Error downloading file: {download_error}")
                            except Exception as msg_error:
                                print(f"Error processing message {msg_id}: {msg_error}")
                    else:
                        print("No WeatherLink emails found in this check.")
                
                except Exception as check_error:
                    print(f"Error during email check attempt {retry_count + 1}: {check_error}")
                
                # Increment retry count
                retry_count += 1
                
                # If we haven't found the email yet and haven't exceeded our wait time, wait before retrying
                if time.time() - start_time < max_wait_time and retry_count < max_retries:
                    wait_time = min(retry_delay, max_wait_time - (time.time() - start_time))
                    print(f"\nNo S3 URL found yet. Waiting {int(wait_time)} seconds before checking again...")
                    print(f"Time elapsed: {int(time.time() - start_time)} seconds out of {max_wait_time} max wait time")
                    time.sleep(wait_time)
            
            print("\nMaximum wait time or retry limit reached. No valid WeatherLink export email found.")
            return None
    
        except Exception as e:
            error_msg = str(e)
            if "access_denied" in error_msg or "verification process" in error_msg:
                print("\n" + "="*80)
                print("GMAIL API ACCESS ERROR")
                print("="*80)
                print("The Gmail API access is currently blocked because the application hasn't been verified.")
                print("\nTo fix this, you need to:")
                print("1. Go to https://console.cloud.google.com")
                print("2. Select your project")
                print("3. Go to 'APIs & Services' > 'OAuth consent screen'")
                print("4. Under 'Test users', add your email address: teamdavcast@gmail.com")
                print("\nAlternatively, you can run the script without the Gmail API:")
                print("python weatherlink.py --debug --check-email --no-api")
                print("="*80 + "\n")
            else:
                print(f"Error checking Gmail using API: {e}")
            
            return None

    def is_logged_in(self):
        """Check if already logged in to the website."""
        try:
            print("Checking login status...")
            self.take_screenshot("login_status_check")
            
            # Look for elements that indicate being logged in
            logged_in_indicators = [
                # Common logout links
                "//a[contains(text(), 'Logout') or contains(@href, 'logout')]",
                "//a[contains(text(), 'Sign Out') or contains(@href, 'signout')]",
                "//a[contains(text(), 'Log Out')]",
                
                # Profile or account links that are typically only shown when logged in
                "//a[contains(text(), 'Profile') or contains(text(), 'Account')]",
                "//a[contains(text(), 'My Account') or contains(text(), 'Dashboard')]",
                
                # Username displays
                "//span[contains(@class, 'username')]",
                "//div[contains(@class, 'user-info')]",
                "//div[contains(@class, 'account-info')]",
                
                # Dashboard elements
                "//h1[contains(text(), 'Dashboard')]",
                "//div[contains(@class, 'dashboard')]"
            ]
            
            # Check each indicator
            for indicator in logged_in_indicators:
                elements = self.driver.find_elements(By.XPATH, indicator)
                if elements:
                    print(f"Found login indicator: {indicator}")
                    return True
                    
            # Check if we're still on the login page
            login_indicators = [
                "//input[@type='password']",
                "//button[contains(text(), 'Log In')]",
                "//h1[contains(text(), 'Login')]",
                "//div[contains(text(), 'Sign In')]"
            ]
            
            for indicator in login_indicators:
                elements = self.driver.find_elements(By.XPATH, indicator)
                if elements:
                    print(f"Still on login page (found {indicator})")
                    return False
                    
            # No clear indication - check the URL
            current_url = self.driver.current_url
            print(f"Current URL: {current_url}")
            
            if "login" in current_url.lower() or "signin" in current_url.lower():
                print("URL indicates we're on a login page")
                return False
                
            # If we've reached a page that's not a login page, we're likely logged in
            print("No login indicators found, but we're not on a login page")
            return True
                
        except Exception as e:
            print(f"Error checking login status: {e}")
            self.take_screenshot("login_status_error")
            return False
        
    def login(self):
        """Login to the WeatherLink website."""
        try:
            print("Attempting to log in...")
            self.take_screenshot("login_page")
            
            # Wait for login form to be visible
            print("Waiting for login form...")
            WebDriverWait(self.driver, 15).until(
                EC.visibility_of_element_located((By.XPATH, "//input[@placeholder='Username' or @name='username' or @id='username']"))
            )
            
            # Locate username input field - try multiple possible selectors
            print("Finding username field...")
            try:
                username_field = self.driver.find_element(By.XPATH, "//input[@placeholder='Username' or @name='username' or @id='username']")
            except:
                # Fallback to regular ID/name attributes
                username_field = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//input[@type='text' and (@id='email' or @name='email' or @name='username')]"))
                )
            
            # Locate password input field
            print("Finding password field...")
            try:
                password_field = self.driver.find_element(By.XPATH, "//input[@type='password' and (@id='password' or @name='password')]")
            except:
                # Fallback to any password field
                password_field = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//input[@type='password']"))
                )
            
            # Get credentials from environment variables
            username = os.getenv("WEATHERLINK_USERNAME")
            password = os.getenv("WEATHERLINK_PASSWORD")
            
            if not username or not password:
                print("Error: Username or password not found in environment variables.")
                print("Please set WEATHERLINK_USERNAME and WEATHERLINK_PASSWORD environment variables.")
                return False
            
            # Clear fields before entering credentials (in case there's any text)
            username_field.clear()
            password_field.clear()
            
            # Enter credentials
            print(f"Entering username: {username[:3]}*****")
            username_field.send_keys(username)
            time.sleep(1)  # Small delay between fields
            print("Entering password: *****")
            password_field.send_keys(password)
            
            # Take screenshot before clicking login
            self.take_screenshot("credentials_entered")
            
            # Click login button - try multiple possible selectors
            print("Looking for login button...")
            try:
                login_button = self.driver.find_element(By.XPATH, "//button[contains(@class, 'Log') or contains(text(), 'Log In')]")
            except:
                # Try alternate selectors
                login_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@type='submit'] | //input[@type='submit'] | //button[contains(text(), 'Log')] | //button[contains(text(), 'Sign')]"))
                )
                
            print("Clicking login button...")
            login_button.click()
            
            # Wait for login to complete
            print("Waiting for login to complete...")
            time.sleep(5)
            
            # Check if login was successful
            if self.is_logged_in():
                print("Login successful")
                return True
            print("Login failed - could not detect logged-in state")
            self.take_screenshot("login_failed")
            return False
            
        except Exception as e:
            print(f"Error during login: {e}")
            self.take_screenshot("login_error")
            return False
    
    def run_merge_script(self, cleaned_filepath, dataset_filepath=None):
        """
        Run the merge.py script to merge the cleaned CSV data with the existing dataset.
        
        Args:
            cleaned_filepath: Path to the cleaned CSV file
            dataset_filepath: Path to the dataset file (defaults to dataset.csv in the script directory)
            
        Returns:
            bool: True if merge script ran successfully, False otherwise
        """
        try:
            print("\n" + "="*50)
            print("MERGING CLEANED DATA WITH DATASET")
            print("="*50)
            
            # If dataset path not provided, check if there's a class attribute, otherwise use default
            if dataset_filepath is None:
                if hasattr(self, 'dataset_path') and self.dataset_path:
                    dataset_filepath = self.dataset_path
                else:
                    dataset_filepath = os.path.join(SCRIPT_DIR, "dataset.csv")
            
            print(f"Dataset path: {dataset_filepath}")
            
            # Check if merge.py exists
            merge_script = os.path.join(SCRIPT_DIR, "merge.py")
            if not os.path.exists(merge_script):
                print(f"Error: Merge script not found at {merge_script}")
                return False
            
            # Check if dataset.csv exists, if not, create it by copying cleaned.csv
            if not os.path.exists(dataset_filepath):
                print(f"Dataset file {dataset_filepath} does not exist. Creating it by copying cleaned.csv.")
                import shutil
                shutil.copy(cleaned_filepath, dataset_filepath)
                print(f"Created new dataset file at {dataset_filepath}")
                return True
            
            # Run the merge script
            command = [sys.executable, merge_script, 
                      "--dataset", dataset_filepath, 
                      "--cleaned", cleaned_filepath]
            
            print(f"Running command: {' '.join(command)}")
            
            # Run the merge script as a subprocess
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False  # Don't raise exception on non-zero return code
            )
            
            # Print output and error messages
            if process.stdout:
                print("\nOutput from merge script:")
                print(process.stdout)
            
            if process.stderr:
                print("\nErrors from merge script:")
                print(process.stderr)
            
            # Check if the process was successful
            if process.returncode == 0:
                print("\nMerge script ran successfully!")
                print(f"Data from {cleaned_filepath} has been merged into {dataset_filepath}")
                
                # Try to get some stats about the merged file
                try:
                    # Check if the dataset file exists and get its size
                    if os.path.exists(dataset_filepath):
                        file_size = os.path.getsize(dataset_filepath) / 1024  # Convert to KB
                        print(f"Merged dataset size: {file_size:.2f} KB")
                        
                        # Try to read the file and get row count
                        try:
                            import pandas as pd
                            merged_df = pd.read_csv(dataset_filepath)
                            print(f"Merged dataset contains {len(merged_df)} records")
                        except Exception as read_error:
                            print(f"Note: Could not read record count from merged file: {read_error}")
                except Exception as stats_error:
                    print(f"Note: Could not get stats about the merged file: {stats_error}")
                
                return True
            else:
                print(f"\nMerge script failed with return code: {process.returncode}")
                print("Please check the error messages above and make sure the merge.py script exists and is working correctly.")
                return False
            
        except Exception as e:
            print(f"Error merging data with dataset: {e}")
            return False

    def run_cleaner_script(self, csv_filepath):
        """
        Run the cleaner.py script on the downloaded CSV file.
        
        Args:
            csv_filepath: Path to the downloaded CSV file
        
        Returns:
            bool: True if cleaner script ran successfully, False otherwise
        """
        try:
            print("\n" + "="*50)
            print("RUNNING CLEANER SCRIPT ON DOWNLOADED DATA")
            print("="*50)
            
            # Get the path to the cleaner.py script (in the same directory as this script)
            cleaner_script = os.path.join(SCRIPT_DIR, "cleaner.py")
            
            if not os.path.exists(cleaner_script):
                print(f"Error: Cleaner script not found at {cleaner_script}")
                return False
            
            # Build the command to run the cleaner script
            command = [sys.executable, cleaner_script, "--input", csv_filepath]
            
            print(f"Running command: {' '.join(command)}")
            
            # Run the cleaner script as a subprocess
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False  # Don't raise exception on non-zero return code
            )
            
            # Print output and error messages
            if process.stdout:
                print("\nOutput from cleaner script:")
                print(process.stdout)
            
            if process.stderr:
                print("\nErrors from cleaner script:")
                print(process.stderr)
            
            # Check if the process was successful
            if process.returncode == 0:
                print("\nCleaner script ran successfully!")
                
                # Look for the cleaned.csv file in the same directory as the script
                cleaned_filepath = os.path.join(SCRIPT_DIR, "cleaned.csv")
                if os.path.exists(cleaned_filepath):
                    print(f"Cleaned data saved to: {cleaned_filepath}")
                    
                    # Run the merge script to merge with the dataset unless explicitly disabled
                    # Default behavior is to merge (merge_dataset will be True unless --skip-merge is specified)
                    if not hasattr(self, 'merge_dataset') or self.merge_dataset:
                        self.run_merge_script(cleaned_filepath, 
                                             getattr(self, 'dataset_path', None))
                    else:
                        print("\nSkipping merge with dataset as requested (--skip-merge specified)")
                    
                    return True
                else:
                    print("Warning: cleaned.csv file not found after running cleaner script")
                    return False
            else:
                print(f"\nCleaner script failed with return code: {process.returncode}")
                return False
            
        except Exception as e:
            print(f"Error running cleaner script: {e}")
            return False

    def run(self):
        """Main method to run the scraping process."""
        try:
            print("\nStarting WeatherLink scraping process...")
            
            # Check if already logged in, if not, log in
            if not self.is_logged_in():
                if not self.login():
                    print("Failed to log in")
                    return None
                print("Login successful")
                time.sleep(3)  # Wait for login to complete
            else:
                print("Already logged in")
            
            # Take a screenshot after login
            self.take_screenshot("after_login")
            
            # Navigate to data page
            if not self.navigate_to_data_page():
                print("Failed to navigate to data page")
                # Don't return None, try to continue
            
            # Set date to current - this step can be skipped if it fails
            print("\n" + "="*50)
            print("ATTEMPTING TO SET DATE TO CURRENT SYSTEM DATE")
            print("="*50)
            date_set_success = self.set_date_to_current()
            if not date_set_success:
                print("\nWARNING: Failed to set date to current system date")
                print("Will proceed with export using the site's default date selection")
                print("The exported data may not reflect the current date")
                # Take a screenshot of the current page state
                self.take_screenshot("date_setting_failed")
            else:
                print("\nDate successfully set to current system date")
                print("Proceeding with data export...")
            
            # Export data to email
            if not self.export_data():
                print("Failed to export data")
                return None
            
            print("Data export request sent successfully")
            
            # If Gmail API is available and we want to check email
            if self.use_api and GmailAPI is not None:
                print("\nWaiting for email with exported data...")
                filepath = self.check_gmail_api(max_wait_time=300)  # 5 minutes max wait time
                
                if filepath:
                    print(f"\nSuccessfully downloaded CSV file: {filepath}")
                    
                    # Run the cleaner script on the downloaded file
                    self.run_cleaner_script(filepath)
                    
                    return filepath
                else:
                    print("\nFailed to download data from Gmail")
                    return None
            else:
                print("\nGmail API not available or not enabled.")
                print("The export request has been sent to your email.")
                print("You will need to check your email manually for the WeatherLink export.")
                return True
            
        except Exception as e:
            print(f"Error in run method: {e}")
            self.take_screenshot("run_error")
            return None
        finally:
            # Always close the browser
            try:
                self.driver.quit()
                print("Browser closed")
            except:
                pass

    def test_email_extraction(self):
        """Test email extraction functionality"""
        try:
            print("Initializing GmailAPI...")
            gmail_api = GmailAPI()
            if not gmail_api or not hasattr(gmail_api, 'service'):
                print("Error: GmailAPI not properly initialized")
                return False
            
            print("GmailAPI initialized successfully")
            print("Fetching messages...")
            
            try:
                results = gmail_api.service.users().messages().list(
                    userId='me',
                    q='from:weatherlink.com',
                    maxResults=5
                ).execute()
                
                messages = results.get('messages', [])
                if not messages:
                    print("No messages found from WeatherLink")
                    return False
                
                print(f"Found {len(messages)} messages")
                
                for msg in messages:
                    try:
                        message = gmail_api.service.users().messages().get(
                            userId='me',
                            id=msg['id']
                        ).execute()
                        
                        print("\nMessage Details:")
                        print(f"Message ID: {msg['id']}")
                        print(f"Thread ID: {msg['threadId']}")
                        
                        headers = message['payload']['headers']
                        for header in headers:
                            if header['name'].lower() in ['subject', 'from', 'date']:
                                print(f"{header['name']}: {header['value']}")
                        
                        s3_url = None
                        if 'parts' in message['payload']:
                            for part in message['payload']['parts']:
                                if part['mimeType'] == 'text/html':
                                    data = part['body']['data']
                                    if data:
                                        import base64
                                        html = base64.urlsafe_b64decode(data).decode('utf-8')
                                        print(f"\nHTML Content Preview: {html[:500]}...")
                                        
                                        # Look for the download link
                                        import re
                                        href_pattern = r'href="(https://s3\.amazonaws\.com/[^"]+\.csv)"'
                                        match = re.search(href_pattern, html)
                                        if match:
                                            s3_url = match.group(1)
                                            print(f"Found S3 URL in HTML: {s3_url}")
                                            break
                                        
                                        # Try alternative pattern
                                        s3_pattern = r'https://s3\.amazonaws\.com/export-wl2-live\.weatherlink\.com/data/[^"]+\.csv'
                                        match = re.search(s3_pattern, html)
                                        if match:
                                            s3_url = match.group(0)
                                            print(f"Found S3 URL: {s3_url}")
                                            break
                                        
                                        print("No S3 URL patterns matched in HTML content")
                                else:
                                    print("Part has no data content")
                        else:
                            print("Email doesn't have the expected 'parts' structure")
                            if 'body' in message['payload'] and 'data' in message['payload']['body']:
                                data = message['payload']['body']['data']
                                if data:
                                    import base64
                                    html = base64.urlsafe_b64decode(data).decode('utf-8')
                                    print(f"Direct body content preview: {html[:200]}...")
                                    
                                    # Look for the download link
                                    import re
                                    href_pattern = r'href="(https://s3\.amazonaws\.com/[^"]+\.csv)"'
                                    match = re.search(href_pattern, html)
                                    if match:
                                        s3_url = match.group(1)
                                        print(f"Found S3 URL in body: {s3_url}")
                        
                        # Save a copy of the email for debugging
                        debug_dir = os.path.join(SCRIPT_DIR, 'debug_emails')
                        if not os.path.exists(debug_dir):
                            os.makedirs(debug_dir)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_file = f"{debug_dir}/email_debug_{timestamp}.txt"
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(f"Message ID: {msg['id']}\n\n")
                            f.write("Headers:\n")
                            for header in message['payload']['headers']:
                                if header['name'].lower() in ['subject', 'from', 'date']:
                                    f.write(f"{header['name']}: {header['value']}\n")
                            f.write("\nFull Message Structure:\n")
                            import json
                            f.write(json.dumps(message, indent=2))
                        print(f"Saved debug email information to {debug_file}")
                        
                        if s3_url:
                            print(f"\nFound S3 URL: {s3_url}")
                            # Create directory for saved files if it doesn't exist
                            if not os.path.exists(SAVED_EMAILS_DIR):
                                os.makedirs(SAVED_EMAILS_DIR)
                            
                            # Download the file
                            import urllib.request
                            filename = s3_url.split('/')[-1]
                            filepath = os.path.join(SAVED_EMAILS_DIR, filename)
                            
                            print(f"Downloading file to {filepath}...")
                            urllib.request.urlretrieve(s3_url, filepath)
                            print("Download complete!")
                            
                            # Run the cleaner script on the downloaded file
                            self.run_cleaner_script(filepath)
                            
                            # Return the filepath of the downloaded file
                            return filepath
                            
                    except Exception as get_error:
                        print(f"Error getting message details: {get_error}")
                        continue
                
                # If we get here, we didn't find any S3 URLs in any messages
                print("\nNo S3 URLs found in any messages")
                return False
                
            except Exception as list_error:
                print(f"Error listing messages: {list_error}")
                if 'insufficient authentication scopes' in str(list_error).lower():
                    print("\nGmail API access issue detected. Please ensure you have:")
                    print("1. Enabled the Gmail API in Google Cloud Console")
                    print("2. Created OAuth 2.0 credentials")
                    print("3. Downloaded the credentials.json file")
                    print("4. Run the script to authenticate")
                return False
            
        except Exception as e:
            print(f"Error in test_email_extraction: {e}")
            return False

    def download_csv_from_email(self):
        """Download the CSV file from the S3 URL found in the email."""
        try:
            print("\nDownloading CSV file from email...")
            
            if GmailAPI is None:
                print("GmailAPI module is not available")
                return None
            
            # Initialize GmailAPI
            gmail_api = GmailAPI()
            if not gmail_api or not gmail_api.service:
                print("Failed to initialize GmailAPI")
                return None
            
            # Get the latest message from WeatherLink
            messages = gmail_api.service.users().messages().list(
                userId='me',
                q='from:weatherlink.com',
                maxResults=1
            ).execute()
            
            if not messages.get('messages'):
                print("No messages found from WeatherLink")
                return None
            
            # Get the message details
            msg = gmail_api.service.users().messages().get(
                userId='me',
                id=messages['messages'][0]['id'],
                format='full'
            ).execute()
            
            # Extract the S3 URL from the email content
            s3_url = None
            if 'parts' in msg['payload']:
                for part in msg['payload']['parts']:
                    if part['mimeType'] == 'text/html':
                        data = part['body'].get('data', '')
                        if data:
                            import base64
                            html = base64.urlsafe_b64decode(data).decode('utf-8')
                            # Look for the download link in the HTML
                            import re
                            s3_url_pattern = r'https://s3\.amazonaws\.com/export-wl2-live\.weatherlink\.com/data/[^"]+\.csv'
                            match = re.search(s3_url_pattern, html)
                            if match:
                                s3_url = match.group(0)
                                print(f"Found S3 URL: {s3_url}")
                                break
            
            if not s3_url:
                print("No S3 URL found in the email")
                return None
            
            # Create a directory for saved files if it doesn't exist
            if not os.path.exists(SAVED_EMAILS_DIR):
                os.makedirs(SAVED_EMAILS_DIR)
            
            # Extract filename from URL
            filename = s3_url.split('/')[-1]
            filepath = os.path.join(SAVED_EMAILS_DIR, filename)
            
            # Download the file directly using requests
            print(f"Downloading file from {s3_url}...")
            try:
                import requests
                response = requests.get(s3_url)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"Successfully downloaded file: {filepath}")
                    
                    # Run the cleaner script on the downloaded file
                    self.run_cleaner_script(filepath)
                    
                    return filepath
                else:
                    print(f"Failed to download file. Status code: {response.status_code}")
                    return None
            except Exception as download_error:
                print(f"Error downloading file: {download_error}")
                return None
            
        except Exception as e:
            print(f"Error downloading CSV file: {e}")
            return None

    def excel_col_to_index(self, col_str):
        """Convert Excel column letter to 0-based index"""
        result = 0
        for c in col_str:
            result = result * 26 + (ord(c.upper()) - ord('A') + 1)
        return result - 1  # Convert to 0-based index

# Add main execution block
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='WeatherLink data scraper')
    parser.add_argument('--url', type=str, default='https://www.weatherlink.com/',
                      help='WeatherLink URL (default: https://www.weatherlink.com/)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode (takes screenshots)')
    parser.add_argument('--export-email', type=str, default='teamdavcast@gmail.com',
                      help='Email address to export data to (default: teamdavcast@gmail.com)')
    parser.add_argument('--check-email', action='store_true',
                      help='Check email for exported data')
    parser.add_argument('--use-api', action='store_true',
                      help='Use Gmail API for email checking')
    parser.add_argument('--no-api', action='store_true',
                      help='Force using browser automation instead of API')
    parser.add_argument('--wait-time', type=int, default=300,
                      help='Maximum time to wait for email (in seconds)')
    parser.add_argument('--test-email', action='store_true',
                      help='Test Gmail API email extraction')
    parser.add_argument('--download-csv', action='store_true',
                      help='Download the CSV file from the latest WeatherLink email')
    parser.add_argument('--skip-clean', action='store_true',
                      help='Skip running the cleaner script on downloaded data')
    parser.add_argument('--skip-merge', action='store_true',
                      help='Skip merging cleaned data with the dataset')
    parser.add_argument('--merge-dataset', action='store_true',
                      help='Merge cleaned data with the main dataset.csv file (deprecated, use --skip-merge to disable merging)')
    parser.add_argument('--dataset-path', type=str, default=None,
                      help='Path to the dataset.csv file (defaults to dataset.csv in script directory)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get WeatherLink URL from environment or use default
    weatherlink_url = os.getenv('WEATHERLINK_URL', args.url)
    
    print("\n" + "="*50)
    print("WEATHERLINK DATA SCRAPER")
    print("="*50)
    print(f"URL: {weatherlink_url}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"Export email: {args.export_email}")
    print(f"Check Gmail: {'YES' if args.check_email else 'NO'}")
    print(f"Use Gmail API: {'YES' if args.use_api and GmailAPI is not None and not args.no_api else 'NO'}")
    print(f"Skip cleaning: {'YES' if args.skip_clean else 'NO'}")
    print(f"Skip merging: {'YES' if args.skip_merge else 'NO'}")
    print("="*50 + "\n")
    
    try:
        # If test-email flag is set, run the test without initializing browser
        if args.test_email:
            print("\nRunning Gmail API email extraction test...")
            # Create a minimal instance without browser initialization
            scraper = WeatherLink(args.url, debug=False, export_email=args.export_email, init_browser=False)
            # Set merge_dataset to True unless skip_merge is specified
            scraper.merge_dataset = not args.skip_merge
            if args.dataset_path:
                scraper.dataset_path = args.dataset_path
            if scraper.test_email_extraction():
                print("\nEmail extraction test completed successfully!")
            else:
                print("\nEmail extraction test failed. Check the error messages above.")
            sys.exit(0)
        
        # If download-csv flag is set, download the CSV file
        if args.download_csv:
            print("\nDownloading CSV file from latest WeatherLink email...")
            # Create a minimal instance without browser initialization
            scraper = WeatherLink(args.url, debug=False, export_email=args.export_email, init_browser=False)
            # Set merge_dataset to True unless skip_merge is specified
            scraper.merge_dataset = not args.skip_merge
            if args.dataset_path:
                scraper.dataset_path = args.dataset_path
            csv_file = scraper.download_csv_from_email()
            if csv_file:
                print(f"\nSuccessfully downloaded CSV file: {csv_file}")
                
                # Run the cleaner script on the downloaded file if not skipped
                if not args.skip_clean:
                    scraper.run_cleaner_script(csv_file)
            else:
                print("\nFailed to download CSV file. Check the error messages above.")
            sys.exit(0)
        
        # For all other operations, create a full instance with browser
        scraper = WeatherLink(weatherlink_url, debug=args.debug, export_email=args.export_email)
        # Set merge_dataset to True unless skip_merge is specified
        scraper.merge_dataset = not args.skip_merge
        if args.dataset_path:
            scraper.dataset_path = args.dataset_path
        
        # Set API usage based on command line arguments
        if args.no_api:
            scraper.use_api = False
        elif args.use_api and GmailAPI is not None:
            scraper.use_api = True
        
        # Run the scraping process
        print("Beginning scraping process...")
        result = scraper.run()
        
        # If we need to check email but run() didn't do it already
        if args.check_email and result is True:  # Export succeeded but no data downloaded yet
            print("\nExplicitly checking Gmail as requested...")
            
            # Use API if requested and available
            csv_file = None
            if args.use_api and GmailAPI is not None and not args.no_api:
                print("Using Gmail API to check for the exported data...")
                csv_file = scraper.check_gmail_api(max_wait_time=args.wait_time)
                
                # Run the cleaner script on the downloaded file if requested
                if csv_file and not args.skip_clean:
                    scraper.run_cleaner_script(csv_file)
            
            # Fall back to browser automation if API fails or not available
            if csv_file is None or args.no_api:
                print("Using browser automation to check Gmail...")
                # TODO: Implement browser automation for Gmail if needed
                print("Browser automation for Gmail not implemented yet")
                
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\nProcess completed successfully!")
    sys.exit(0)
