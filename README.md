



## Download the Model

The current model is stored on a dropbox link as I could not load it on Github due to its size.

I will provide the required steps to make the download via code.

First navigate to the `code` directory from the repository

Next, download the file
wget https://www.dropbox.com/scl/fo/ezwd61ezc589gpp88u89t/h?rlkey=ckp9bl7269wpvptxywtr7hcrw&dl=1

Wait for the Download to conclude, then rename the downloaded file
mv 'h?rlkey=ckp9bl7269wpvptxywtr7hcrw' ml-zoomcamp-qa-sentence-transformer.zip

Now use the unzip command to extract the files
unzip ml-zoomcamp-qa-sentence-transformer.zip -d ml-zoomcamp-qa-sentence-transformer

if you don't have unzip you can install it with `sudo apt install unzip`

Cleaning the not necessary files
rm ml-zoomcamp-qa-sentence-transformer.zip && rm  wget-log

You can make the download pasting the link on a browser
