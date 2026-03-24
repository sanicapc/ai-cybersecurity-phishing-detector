import tldextract

def extract_url_features(url):
    ext = tldextract.extract(url)
    
    features = {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "has_https": 1 if "https" in url else 0,
        "num_subdomains": len(ext.subdomain.split('.')) if ext.subdomain else 0
    }
    
    return list(features.values())
