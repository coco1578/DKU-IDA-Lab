# Tor browser sniffer

### Original version's of the crawler

[**origianl version**](https://github.com/webfp/tor-browser-crawler)

Read the original version's **configure the environment**

------

#### Requirement

###### 	Program

- geckodriver 0.17.0 version
  - [geckodriver](https://github.com/mozilla/geckodriver/releases/download/v0.17.0/geckodriver-v0.17.0-linux64.tar.gz)
- tor browser bundle 7.5.4 version
  - [tor browser](https://archive.torproject.org/tor-package-archive/torbrowser/7.5.4/tor-browser-linux64-7.5.4_en-US.tar.xz)
    - unzip the tor-browser-linux64-7.5.4_en-US.tar.xz file in the Sniffer/tbb/tor-browser-linux64-7.5.4_en-US/ folder

###### 	Python module

- selenium
- stem
- tld

------

#### Usage

```python
python3 -u url.txt
```