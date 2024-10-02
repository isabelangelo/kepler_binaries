from specmatchemp.spectrum import read_hires_fits
import pexpect

# load original wavelength data
original_wav_file = read_hires_fits('./data/cks-spectra/rj122.742.fits') # KOI-1 original r chip file
original_wav_data = original_wav_file.w[:,:-1] # require even number of elements

def run_rsync(command):
	"""
	Function to run general terminal command, automatically logging 
	into caltech's cadence server. 
	"""
	# run the command using pexpect
	program = pexpect.spawn(command)
	program.expect("observer@cadence.caltech.edu's password:")
	program.sendline("CPS<3RVs")
	program.expect(pexpect.EOF)