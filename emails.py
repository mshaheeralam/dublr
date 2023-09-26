import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()

email_message = {
    'success': "<p style='color : black;font-size: 18px';>Your video <strong>{}</strong> has been dubbed successfully and is now available in your library.</p><p style='color: black font-size: 14px'>You can access it right away by logging into your account on our website.</p>",
    'error': "<p style='color : black;font-size: 18px';>Your video <strong>{}</strong> was not dubbed successfully.</p><p style='color: black font-size: 14px'>The dubbing time required for this video has been refunded to your account.</p>",
    'multi_speaker_error': "<p style='color : black;font-size: 18px';>Processing failed for <strong>{}</strong> because the video contains multiple speakers.</p><p style='color: black font-size: 14px'>Please ensure the video you provide contains only one speaker and try again.<br>The dubbing time required for this video has been refunded to your account.</p>",
    'segmentation_error': "<p style='color : black;font-size: 18px';>Processing failed for <strong>{}</strong> because the video does not contain audio.</p><p style='color: black font-size: 14px'>Please ensure the video you provide contains audio and try again.<br><br>The dubbing time required for this video has been refunded to your account.</p>"
}

# ... rest of your code

def send_email(status, video_name ,RECIPIENT):
    if status == "success":
        link_text="View Result"
    else:
        link_text="Try Again"

    client = boto3.client('ses')
    SENDER = "info@dublr.ai"
    SUBJECT = "Video Dubbing Status"
    BODY_HTML = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: white;
            color: black;
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 800px;
            margin: auto;
            background-color: white;
            padding: 20px;
            text-align: center;
        }}
        @media only screen and (max-width: 600px) {{
            .container {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class='container'>
        <img src='https://www.dublr.ai/favicon.ico' alt='Dublr' style='max-width: 200px;'>
        <h1 style='color : black;font-size: 24px; margin-bottom: 20px;'>Video Processing | Update</h1>
        <p style='color : black;font-size: 16px; line-height: 1.5;'>{email_message[status].format(video_name)}</p>
        <a href='https://www.dublr.ai/#dublr-feedback-zone' style='text-decoration: none; background-color: black; color: white; padding: 10px 20px; border-radius: 5px; margin-top: 20px; display: inline-block;'>{link_text}</a>
    </div>
</body>
</html>
"""

    CHARSET = "UTF-8"
    try:
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER
        )
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:", response['MessageId'])


# status="error"
# video_name="My_video"
# rec="aliagha135@hotmail.com"
# send_email(status, video_name, RECIPIENT=rec)