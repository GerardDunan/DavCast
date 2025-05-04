import os
import base64
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import pandas as pd
from datetime import datetime
import time
import re
import requests
import socket
import sys
from google.oauth2.credentials import Credentials

class GmailAPI:
    """Class to interact with Gmail API to fetch emails and download attachments."""
    
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        self.service = self.get_gmail_service()
        
    def get_gmail_service(self):
        """Get or create Gmail API service."""
        creds = None
        
        # Check if token.pickle exists in the weatherlink folder
        token_path = os.path.join(os.path.dirname(__file__), 'token.pickle')
        credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
        
        # Check if token.pickle exists
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)

        # If credentials are invalid or don't exist, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(credentials_path):
                    print("Error: credentials.json not found!")
                    print("Please download your OAuth 2.0 credentials from Google Cloud Console")
                    print(f"and save them as 'credentials.json' in the weatherlink folder: {os.path.dirname(__file__)}")
                    return None

                # Create flow with offline access
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path,
                    self.SCOPES
                )

                # Set the redirect URI before getting the authorization URL
                flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

                # Get the authorization URL
                auth_url, _ = flow.authorization_url(
                    access_type='offline',
                    prompt='consent'
                )

                print("\n" + "="*80)
                print("MANUAL AUTHENTICATION REQUIRED")
                print("="*80)
                print("1. Visit this URL in your web browser:")
                print("\n" + auth_url + "\n")
                print("2. Sign in with your Google account and grant the requested permissions")
                print("3. After approval, you'll see a code on the page")
                print("4. Copy that code and paste it here")
                print("="*80 + "\n")

                # Get the authorization code from user input
                code = input("Enter the authorization code: ").strip()
                
                try:
                    # Exchange the authorization code for credentials
                    flow.fetch_token(code=code)
                    creds = flow.credentials
                    print("Successfully obtained credentials!")
                except Exception as e:
                    print(f"Error exchanging authorization code: {e}")
                    return None

            # Save the credentials for future use in the weatherlink folder
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
            print(f"Saved credentials to: {token_path}")

        try:
            # Build and return the Gmail service
            service = build('gmail', 'v1', credentials=creds)
            return service
        except Exception as e:
            print(f"Error building Gmail service: {e}")
            return None
    
    def get_messages(self, query="from:weatherlink.com", max_results=10):
        """
        Get messages matching a search query.
        
        Args:
            query: Gmail search query
            max_results: Maximum number of results to return
            
        Returns:
            list: List of message objects
        """
        try:
            if not self.service:
                print("Gmail service not initialized, attempting to authenticate...")
                if not self.get_gmail_service():
                    print("Authentication failed, cannot get messages")
                    return []
            
            print(f"Searching for messages with query: {query}")
            
            # Get list of messages
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            print(f"Found {len(messages)} messages")
            return messages
            
        except Exception as e:
            print(f"Error getting messages: {e}")
            # If authentication error, try to re-authenticate
            if "401" in str(e) or "unauthorized" in str(e).lower():
                print("Authentication error detected, attempting to re-authenticate...")
                if os.path.exists(os.path.join(os.path.dirname(__file__), 'token.pickle')):
                    os.remove(os.path.join(os.path.dirname(__file__), 'token.pickle'))
                    print(f"Removed token file: {os.path.join(os.path.dirname(__file__), 'token.pickle')}")
                if self.get_gmail_service():
                    print("Re-authentication successful, retrying query...")
                    return self.get_messages(query, max_results)
            return []
    
    def get_message_content(self, message_id):
        """
        Get the content of a specific message.
        
        Args:
            message_id: ID of the message to get
            
        Returns:
            dict: Message content object
        """
        try:
            if not self.service:
                if not self.get_gmail_service():
                    return None
            
            message = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()
            
            return message
            
        except Exception as e:
            print(f"Error getting message content: {e}")
            return None
    
    def download_attachment(self, message_id, attachment_id, attachment_name):
        """
        Download an attachment from a message.
        
        Args:
            message_id: ID of the message containing the attachment
            attachment_id: ID of the attachment to download
            attachment_name: Name to save the attachment as
            
        Returns:
            str: Path to the downloaded attachment or None if download failed
        """
        try:
            if not self.service:
                if not self.get_gmail_service():
                    return None
            
            attachment = self.service.users().messages().attachments().get(
                userId='me',
                messageId=message_id,
                id=attachment_id
            ).execute()
            
            # Decode the attachment data
            file_data = base64.urlsafe_b64decode(attachment['data'].encode('UTF-8'))
            
            # Generate a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{attachment_name.split('.')[0]}_{timestamp}.csv"
            
            # Save the attachment
            with open(output_filename, 'wb') as f:
                f.write(file_data)
            
            print(f"Attachment saved as: {output_filename}")
            return output_filename
            
        except Exception as e:
            print(f"Error downloading attachment: {e}")
            return None
    
    def find_weatherlink_export_and_download(self, max_wait_time=300):
        """
        Find the latest WeatherLink export email and download the CSV file from the provided link.
        
        Args:
            max_wait_time: Maximum time to wait for the email (in seconds)
            
        Returns:
            str: Path to the downloaded CSV file or None if not found
        """
        try:
            print("\nLooking for WeatherLink export emails...")
            start_time = time.time()
            retry_count = 0
            max_retries = 10
            retry_delay = 30  # seconds
            
            # Check for emails multiple times with delays
            while time.time() - start_time < max_wait_time and retry_count < max_retries:
                retry_count += 1
                print(f"\nAttempt {retry_count}/{max_retries} to find WeatherLink export email...")
                
                # Search for messages from WeatherLink
                messages = self.get_messages(
                    query="from:weatherlink.com",
                    max_results=5
                )
                
                if messages:
                    print(f"Found {len(messages)} messages. Processing the latest one...")
                    # Get the most recent message (first in the list)
                    latest_message = self.get_message_content(messages[0]['id'])
                    
                    if latest_message and 'payload' in latest_message:
                        # Get the email body
                        if 'parts' in latest_message['payload']:
                            for part in latest_message['payload']['parts']:
                                if part['mimeType'] == 'text/html':
                                    # Decode the email body
                                    body_data = part.get('body', {}).get('data', '')
                                    if not body_data:
                                        print("Email body data is empty")
                                        continue
                                        
                                    body = base64.urlsafe_b64decode(body_data).decode('utf-8')
                                    
                                    # Try multiple patterns to find the download link
                                    download_patterns = [
                                        r'href="(https://s3\.amazonaws\.com/export-wl2-live\.weatherlink\.com/data/[^"]+\.csv)"',
                                        r'href="(https://s3\.amazonaws\.com/[^"]+\.csv)"',
                                        r'(https://s3\.amazonaws\.com/export-wl2-live\.weatherlink\.com/data/[^"]+\.csv)',
                                        r'(https://s3\.amazonaws\.com/[^"]+\.csv)'
                                    ]
                                    
                                    download_url = None
                                    for pattern in download_patterns:
                                        download_link_match = re.search(pattern, body)
                                        if download_link_match:
                                            download_url = download_link_match.group(1)
                                            print(f"Found download link with pattern: {pattern}")
                                            break
                                    
                                    if download_url:
                                        print(f"Found download link: {download_url}")
                                        
                                        # Create directory for downloads if it doesn't exist
                                        downloads_dir = 'weather_downloads'
                                        if not os.path.exists(downloads_dir):
                                            os.makedirs(downloads_dir)
                                        
                                        # Download the file
                                        try:
                                            print(f"Downloading file from {download_url}...")
                                            response = requests.get(download_url, timeout=30)
                                            if response.status_code == 200:
                                                # Generate a filename with timestamp
                                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                filename = os.path.join(downloads_dir, f"weather_data_{timestamp}.csv")
                                                
                                                # Save the file
                                                with open(filename, 'wb') as f:
                                                    f.write(response.content)
                                                
                                                print(f"File downloaded successfully: {filename}")
                                                
                                                # Also save as dataset.csv in current directory
                                                with open("dataset.csv", 'wb') as f:
                                                    f.write(response.content)
                                                print("Also saved as dataset.csv in the current directory")
                                                
                                                return filename
                                            else:
                                                print(f"Failed to download file. Status code: {response.status_code}")
                                        except Exception as download_error:
                                            print(f"Error downloading file: {download_error}")
                                    else:
                                        print("No download link found in the email")
                        else:
                            print("No parts found in the email payload")
                    else:
                        print("Could not get message content or invalid format")
                else:
                    print("No messages found from WeatherLink")
                
                # If we haven't found and downloaded the file yet, wait before retrying
                if time.time() - start_time < max_wait_time and retry_count < max_retries:
                    wait_time = min(retry_delay, max_wait_time - (time.time() - start_time))
                    print(f"\nWaiting {int(wait_time)} seconds before trying again...")
                    print(f"Time elapsed: {int(time.time() - start_time)} seconds out of {max_wait_time} seconds maximum wait time")
                    time.sleep(wait_time)
            
            print(f"\nReached maximum wait time or retry limit. No WeatherLink export email with download link found.")
            return None
            
        except Exception as e:
            print(f"Error finding and downloading WeatherLink export: {e}")
            return None

# Add test functionality for direct script execution
if __name__ == "__main__":
    print("Gmail API Test Script")
    print("=====================")
    print("This script tests the Gmail API authentication and WeatherLink email search functionality.")
    
    # Check if credentials file exists
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'credentials.json')):
        print("ERROR: credentials.json file not found!")
        print("Please download OAuth credentials from Google Cloud Console")
        sys.exit(1)
    
    print("\nInitializing Gmail API...")
    api = GmailAPI()
    
    if api.service:
        print("\nAUTHENTICATION SUCCESSFUL!")
        
        # Ask user if they want to search for emails
        search = input("\nWould you like to search for WeatherLink emails? (y/n): ").lower() == 'y'
        if search:
            print("\nSearching for WeatherLink emails...")
            result = api.find_weatherlink_export_and_download(max_wait_time=60)  # 1 minute for testing
            if result:
                print(f"\nSuccess! Downloaded file: {result}")
            else:
                print("\nNo WeatherLink emails with download links found.")
    else:
        print("\nAuthentication failed. Please check the error messages above.")
        sys.exit(1) 
