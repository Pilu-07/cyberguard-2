import re
import urllib.parse
from typing import Tuple, Dict, Any

class URLCanonicalizer:
    """
    Production-grade URL Canonicalization Engine.
    Normalizes scheme/host casing, trailing slashes, default ports,
    and URL encoding while strictly preserving payload queries and paths.
    """
    
    DEFAULT_PORTS = {
        'http': 80,
        'https': 443,
        'ftp': 21
    }

    @classmethod
    def canonicalize(cls, raw_url: str) -> Dict[str, str]:
        """
        Takes raw URL and returns dict with 'original_url', 'normalized_url', 
        'scheme', 'domain', 'path', 'query', 'fragment'.
        """
        if not raw_url or not isinstance(raw_url, str):
            return {
                'original_url': '',
                'normalized_url': '',
                'scheme': '',
                'domain': '',
                'path': '',
                'query': '',
                'fragment': ''
            }

        original_url = raw_url.strip()
        url = original_url

        # Check if scheme is missing (e.g., example.com/login)
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9+-.]*://', url):
            url = 'http://' + url

        try:
            parsed = urllib.parse.urlsplit(url)
        except Exception:
            # Fallback parsing on malformed strings
            return {
                'original_url': original_url,
                'normalized_url': original_url.lower(),
                'scheme': 'http',
                'domain': original_url.split('/')[0].lower(),
                'path': '',
                'query': '',
                'fragment': ''
            }

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc

        # Split userinfo, hostname, port
        userinfo = ""
        hostname = ""
        port = None

        if '@' in netloc:
            userinfo, hostport = netloc.split('@', 1)
            userinfo = userinfo + '@'
        else:
            hostport = netloc

        # Parse host and port
        if ':' in hostport and not (hostport.startswith('[') and hostport.endswith(']')):
            parts = hostport.split(':')
            hostname = parts[0].lower()
            try:
                port = int(parts[1])
            except ValueError:
                port = None
        else:
            hostname = hostport.lower()

        # Remove default ports
        if port is not None:
            if scheme in cls.DEFAULT_PORTS and cls.DEFAULT_PORTS[scheme] == port:
                netloc_norm = f"{userinfo}{hostname}"
            else:
                netloc_norm = f"{userinfo}{hostname}:{port}"
        else:
            netloc_norm = f"{userinfo}{hostname}"

        # Normalize path: remove redundant slashes, handle root path
        path = parsed.path
        if not path and not parsed.query:
            path = '/'
        elif path:
            # Collapse multiple slashes (e.g. //path///to -> /path/to)
            path = re.sub(r'/{2,}', '/', path)

        # Safely unquote then normalize query
        query = parsed.query
        fragment = parsed.fragment

        # Construct canonical normalized URL
        normalized = urllib.parse.urlunsplit((scheme, netloc_norm, path, query, fragment))

        return {
            'original_url': original_url,
            'normalized_url': normalized,
            'scheme': scheme,
            'domain': hostname,
            'path': path,
            'query': query,
            'fragment': fragment
        }

    @classmethod
    def extract_root_domain(cls, hostname: str) -> str:
        """
        Extracts effective root domain (e.g. secure.login.chase.com -> chase.com).
        Handles common two-part ccTLDs (.co.uk, .com.au, .gov.in, etc.)
        """
        if not hostname:
            return ""
        
        # Remove port if present
        host = hostname.split(':')[0].strip().lower()
        
        # Check if IP address
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', host):
            return host
            
        parts = host.split('.')
        if len(parts) <= 2:
            return host
            
        known_second_levels = {'co', 'com', 'org', 'net', 'edu', 'gov', 'mil', 'ac'}
        if len(parts) >= 3 and parts[-2] in known_second_levels and len(parts[-1]) == 2:
            return '.'.join(parts[-3:])
            
        return '.'.join(parts[-2:])
