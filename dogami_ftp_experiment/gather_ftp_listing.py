server = "lidar.engr.oregonstate.edu"
basedir = "OREGON LIDAR CONSORTIUM PROJECT DATA"
datafolder = "BARE EARTH"

import ftplib
import argparse
import json
import logging
import re
from collections import defaultdict
from urllib.parse import quote
import time


def connect_ftp(host, user=None, passwd=None, timeout=30):
	ftp = ftplib.FTP()
	ftp.connect(host, timeout=timeout)
	if user:
		ftp.login(user, passwd or "")
	else:
		ftp.login()
	return ftp


def safe_nlst(ftp):
	try:
		return ftp.nlst()
	except ftplib.error_perm:
		# some FTP servers return 550 for empty dirs
		return []
	except Exception:
		return []


def list_dir(ftp):
	"""Return a list of (name, is_dir) for the current ftp working directory.
	Prefer MLSD (structured), fall back to NLST + probe.
	"""
	try:
		entries = list(ftp.mlsd())
		res = []
		for name, facts in entries:
			if name in ('.', '..'):
				continue
			typ = facts.get('type', '').lower()
			is_dir = typ == 'dir'
			res.append((name, is_dir))
		return res
	except (AttributeError, ftplib.error_perm, ftplib.error_temp, OSError):
		# MLSD not supported or failed; fall back
		names = safe_nlst(ftp)
		res = []
		for n in names:
			if n in ('.', '..'):
				continue
			res.append((n, is_directory(ftp, n)))
		return res


def is_directory(ftp, name):
	"""Return True if `name` is a directory in the current ftp cwd."""
	cur = ftp.pwd()
	try:
		ftp.cwd(name)
		ftp.cwd(cur)
		return True
	except Exception:
		# not a directory or inaccessible
		try:
			ftp.cwd(cur)
		except Exception:
			pass
		return False


def make_ftp_uri(server, cwd, name):
	# build ftp://server/<quoted path>
	full = cwd.rstrip('/') + '/' + name
	parts = [quote(p) for p in full.split('/') if p != '']
	return f'ftp://{server}/' + '/'.join(parts)


def find_be_dirs_under(ftp, index, server, parent_index=None, be_pattern=re.compile(r'^be\d{4}[A-Za-z]\d', re.I)):
	"""Recursively find directories that start with 'be' under current cwd and add to index.
	Uses structured listings when available and logs progress.
	"""
	entries = list_dir(ftp)
	found_any = False
	for e, is_dir in entries:
		if not is_dir:
			continue
		logging.debug('Inspecting for BE dir: %s (cwd=%s)', e, ftp.pwd())
		if be_pattern.match(e):
			uri = make_ftp_uri(server, ftp.pwd(), e)
			ohio = e[2:]
			# avoid duplicate identical URIs
			if uri not in index[ohio]:
				index[ohio].append(uri)
				logging.info('Found be dir: %s -> %s', e, uri)
			# also record under the canonical parent (first matched portion)
			if parent_index is not None:
				m = be_pattern.match(e)
				if m:
					parent_key = m.group(0)[2:]
					if uri not in parent_index[parent_key]:
						parent_index[parent_key].append(uri)
			found_any = True
			# no need to recurse further into a be folder
			continue

		# recurse into subdirectory
		try:
			ftp.cwd(e)
			logging.debug('Entering %s', ftp.pwd())
			# if recursion finds any, mark found_any
			if find_be_dirs_under(ftp, index, server, parent_index, be_pattern):
				found_any = True
			ftp.cwd('..')
		except KeyboardInterrupt:
			raise
		except Exception:
			logging.debug('Could not enter or recurse into %s', e)
			try:
				ftp.cwd('..')
			except Exception:
				pass

	return found_any


def traverse_and_index(ftp, server, basedir, datafolder):
	"""Traverse from `basedir`, locate directories containing `datafolder` in their name
	and index any `be*` subdirectories found under them.
	Returns a dict mapping ohio-grid -> list of full ftp paths.
	"""
	index = defaultdict(list)
	parent_index = defaultdict(list)
	try:
		ftp.cwd(basedir)
	except Exception as e:
		raise RuntimeError(f"Could not change directory to {basedir}: {e}")

	stop_list = set(['3dep', 'data_report', 'data report', 'report', 'points', 'highest_hit', 'intensity', 'vector', 'vectors'])

	def normalize(name):
		n = name.lower().replace('_', ' ').strip()
		# collapse multiple spaces
		n = re.sub(r'\s+', ' ', n)
		return n

	def _recurse():
		found_here = False
		entries = list_dir(ftp)
		for e, is_dir in entries:
			if not is_dir:
				continue

			n = normalize(e)
			logging.debug('At %s: found directory %s (norm=%s)', ftp.pwd(), e, n)

			# if directory name is in the stop list, skip descending this branch
			if n in stop_list:
				logging.info('Skipping stop-list directory: %s', e)
				continue

			# if this directory is a raster folder (raster/rasters), look only for exact 'bare earth' children
			if 'raster' in n:
				try:
					ftp.cwd(e)
				except Exception:
					logging.debug('Could not enter raster directory %s', e)
					continue

				logging.debug('Scanning inside raster dir: %s', ftp.pwd())
				inner = list_dir(ftp)
				for ie, ie_is_dir in inner:
					if not ie_is_dir:
						continue
					inorm = normalize(ie)
					# match only exact 'bare earth'
					if inorm == 'bare earth':
						try:
							ftp.cwd(ie)
						except Exception:
							logging.debug('Could not enter bare earth dir %s', ie)
							continue
						logging.info('Scanning datafolder candidate: %s', ftp.pwd())
						if find_be_dirs_under(ftp, index, server, parent_index):
							found_here = True
						try:
							ftp.cwd('..')
						except Exception:
							pass

				try:
					ftp.cwd('..')
				except Exception:
					pass
				# done with this raster branch
				continue

			# otherwise descend normally
			try:
				ftp.cwd(e)
			except Exception:
				logging.debug('Skipping unreadable directory %s', e)
				continue

			if _recurse():
				found_here = True

			# go back up
			try:
				ftp.cwd('..')
			except Exception:
				pass

		if not found_here:
			logging.warning('No be* folders found under %s (consider adding to stop-list)', ftp.pwd())
		return found_here

	try:
		_recurse()
	except KeyboardInterrupt:
		logging.warning('Traversal interrupted by user; returning partial index')
		return index

	return index, parent_index


def main():
	parser = argparse.ArgumentParser(description='Index be* ohio-grid folders on a remote FTP server')
	parser.add_argument('--server', default=server, help='FTP server hostname')
	parser.add_argument('--basedir', default=basedir, help='Base directory on the FTP server to scan')
	parser.add_argument('--datafolder', default=datafolder, help='Phrase identifying the data folder (e.g. "BARE EARTH")')
	parser.add_argument('--user', help='FTP username (omit for anonymous)')
	parser.add_argument('--password', help='FTP password')
	parser.add_argument('--out', help='Output JSON file (default: stdout)')
	parser.add_argument('--dry-run', action='store_true', help='Do not connect; just print settings')
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	if args.dry_run:
		print(json.dumps(dict(server=args.server, basedir=args.basedir, datafolder=args.datafolder), indent=2))
		return

	ftp = connect_ftp(args.server, args.user, args.password)
	try:
		logging.info('Connected to %s, scanning %s', args.server, args.basedir)
		start = time.time()
		idx, parent_idx = traverse_and_index(ftp, args.server, args.basedir, args.datafolder)
		duration = time.time() - start
		logging.info('Scan finished in %.1fs', duration)
	finally:
		try:
			ftp.quit()
		except Exception:
			pass

	# normalize index -> lists and sort for both maps
	result = {
		"subgrids": {k: sorted(v) for k, v in idx.items()},
		"parents": {k: sorted(v) for k, v in parent_idx.items()},
	}

	if args.out:
		with open(args.out, 'w') as fh:
			json.dump(result, fh, indent=2)
		logging.info('Wrote index to %s', args.out)
	else:
		print(json.dumps(result, indent=2))


if __name__ == '__main__':
	main()