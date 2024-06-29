
ChordBRR
Version 0.80b
by Dzing

-------------------------------

Table of Contents

 1) What is ChordBRR
	1.1) Just intonation
	1.2) 12 note equal temperament
	1.3) Beating
 2) Installation
 3) How to use
	3.1) Selection of BRR file (Tab 1)
	3.2) Selection of notes in chord (Tab 2)
	3.3) Finalizing the sample (Tab 3)
 4) Known issues

1) What is ChordBRR

ChordBRR is a tool that uses existing looped BRR-files and produces wav files that can play several notes at the same time.

The tool uses approximations when calculating the different chords. Since frequencies of notes (in equal temperament) do not perfectly line up, the tool stretches the individual waves in order to achieve the smallest possible size within a given error range.
It is recommended for users to familiarize themselves with the musical concepts described below before using the tool.
This youtube video describes these concepts well: https://www.youtube.com/watch?v=7JhVcGtT8z4

1.1) Just intonation

When playing a string on for example a guitar, this string can vibrate on its whole length producing the frequency that it was tuned to. However, vibrations of 2, 3, 4 etc. can also occur. These vibrations are called the harmonic series, they happen naturally and are conceived as pleasant to listen to.
Tuning to these harmonic series in music is called just (pure) intonation tuning. One note is chosen as the base and all other notes frequencies are given as fractions of this note.

1.2) 12 note equal temperament

In modern music just intonation is seldom used. Our modern system of tuning, called 12 note equal temperament, is a compromise. We divide the octave into 12 equal intervals not because it sound better that way, but so we can transpose any music to any key.
The main problem with just intonation tuning is that if all notes are tuned to a certain note for example C. Chords within the scale of this sound great. However, changing the scale to for example a base D, the chords will sound awful.
With instruments like for example the piano, it is impossible to change the tuning constantly while playing. Equal temperament is used in order for all scales to sound equally good (or bad).
The table below shows the error compared to just intonation tuning for all intervals.

Errors from the equal temperament tuning (in cents):
Octave			0
Perfect Fifth	+1.96
Perfect Fourth	-1.96
Major Third		-13.69
Minor Third		+15.64
Major Sixth		-15.64
Minor Sixth		+13.69
Major Seventh	-11.73
Minor Seventh	+17.6
Tritone			-17.49
Major Second	+3.91
Minor Second	+11.73

1.3) Beating

In acoustics, a beat is an interference pattern between two sounds of slightly different frequencies, perceived as a periodic variation in volume whose rate is the difference of the two frequencies.

With tuning instruments that can produce sustained tones, beats can be readily recognized. Tuning two tones to a unison will present a peculiar effect: when the two tones are close in pitch but not identical, the difference in frequency generates the beating.
The volume varies like in a tremolo as the sounds alternately interfere constructively and destructively. As the two tones gradually approach unison, the beating slows down and may become so slow as to be imperceptible.

2) Installation

This tool requires python v3.x, the tool should run on windows, linux and mac (tested only on windows)

The tool has the following dependencies (installed using "pip install"):
numpy
dearpygui
scipy
sounddevice

The tool can be run with the command line "python chordbrr.py"


3) How to use

The UI of the tool consists of 3 different pages (tabs) which you can swap between using the back and next buttons. When opening the tool, these buttons will be disabled until a BRR file is loaded.
The header shows information about the current loaded BRR file.
Tuning:
The tuning of the BRR sample. Currently this only affects the playback of the sample and the frequency of the output wav. Changing the tuning does not affect the final BRR file in any way.

3.1) Selection of BRR file (Tab 1)
Open BRR file:
Lets you open a BRR file. Currently the BRR files need to have the correct format with a 2 bytes header followed by blocks of 9 bytes for nibble data.
In theory all looped samples can be used, but samples with large loops can result in large BRR files. It is recommended to use samples that have only one wave as the loop.
The tool will use the original sample as base for the lowest note in the chord and will add waves with higher frequencies on top.
This means that in cases where the original sample has a low sample rate, there will be a major loss in quality of the output chord sample. In these cases it is recommended to resample the BRR file with a higher sample rate.

3.2) Selection of notes in chord (Tab 2)
Number of notes:
Select the number of notes in the chord (up to 5) and the individual notes in the chord. Note that chords with the same intervals between notes will produce the same sample (for example a sample with the notes C and D will be the same as a sample with D and E).

Error threshold:
Choose the maximum error allowed in your chord. The list below shows the matches found below the error threshold. Which lets you choose the desired output (accuracy vs size).
As a general rule if you keep the error below 5 cent, it should be hard to notice the difference in most cases. Humans tend to be somewhat forgiving when it comes to tuning in chords as we are used to the 12 note equal temperament tuning (which is out of tune).
One should be aware of beating which typically occurs when playing the same note with slightly different tuning. If this occurs it might be possible to accommodate for it using the $EE command for these notes.

Just intonation mode:
Only allows for matches going towards just intonation. When choosing this mode and an error greater than 18 cent, the smallest match will be perfectly in just intonation tuning.

3.3) Finalizing the sample (Tab 3)

Choose the volume and delay (in samples) for each note in the chord.

Play:
Plays the current chord (tuning of original sample is required for this)

Save:
Saves the sample as a wav file.
The output wav file can be converted with a conversion tool. The loop point is stored in the wav file.
The resulting brr will have the same tuning as the original sample and should be played as the lowest note in your chord.

4) Known issues

- There are no error messages when loading a BRR file of the wrong format
- There is no check whether a BRR sample is looped or not and the tool assumes the sample is looped


Things I would like to implement:
- Loading of "!pattern.txt" files to automatically load tuning
- Option for resampling the BRR sample
- Show error for each individual note
- Possibly, the loading of multiple samples
