/*
 * FilterViews.cpp
 *
 * Copyright (c) 2014-2015 SEACAVE
 *
 * Author(s):
 *
 *      cDc <cdc.seacave@gmail.com>
 *      Romain Janvier <romain.janvier~AT~univ-orleans.fr>
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * Additional Terms:
 *
 *      You are required to preserve legal notices and author attributions in
 *      that material or in the Appropriate Legal Notices displayed by works
 *      containing it.
 */

#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"
#include <boost/program_options.hpp>
#include <boost/regex.hpp>

using namespace MVS;

// D E F I N E S ///////////////////////////////////////////////////

#define APPNAME _T("FilterViews")


// S T R U C T S ///////////////////////////////////////////////////

namespace OPT {
String strInputFileName;
String strRegexExp;
String strOutputFileName;
String strConfigFileName;
unsigned nArchiveType;
int nProcessPriority;
unsigned nMaxThreads;
boost::program_options::variables_map vm;
} // namespace OPT

// initialize and parse the command line parameters
bool Initialize(size_t argc, LPCTSTR* argv)
{
  // initialize log and console
  OPEN_LOG();
  OPEN_LOGCONSOLE();

  // group of options allowed only on command line
  boost::program_options::options_description generic("Generic options");
  generic.add_options()
          ("help,h", "produce this help message")
          ("working-folder,w", boost::program_options::value<std::string>(&WORKING_FOLDER), "working directory (default current directory)")
          ("config-file,c", boost::program_options::value<std::string>(&OPT::strConfigFileName)->default_value(APPNAME _T(".cfg")), "file name containing program options")
          ("archive-type", boost::program_options::value<unsigned>(&OPT::nArchiveType)->default_value(2), "project archive type: 0-text, 1-binary, 2-compressed binary")
          ("process-priority", boost::program_options::value<int>(&OPT::nProcessPriority)->default_value(-1), "process priority (below normal by default)")
          ("max-threads", boost::program_options::value<unsigned>(&OPT::nMaxThreads)->default_value(0), "maximum number of threads (0 for using all available cores)")
          #if TD_VERBOSE != TD_VERBOSE_OFF
          ("verbosity,v", boost::program_options::value<int>(&g_nVerbosityLevel)->default_value(
            #if TD_VERBOSE == TD_VERBOSE_DEBUG
            3
            #else
            2
            #endif
            ), "verbosity level")
          #endif
          ;

  // group of options allowed both on command line and in config file
  boost::program_options::options_description config("Refine options");
  config.add_options()
          ("input-file,i", boost::program_options::value<std::string>(&OPT::strInputFileName), "input filename containing camera poses and image list")
          ("output-file,o", boost::program_options::value<std::string>(&OPT::strOutputFileName), "output filename for storing the mesh")
          ("regex-epression,r", boost::program_options::value<std::string>(&OPT::strRegexExp)->default_value(".*"), "regex expression to filter views/images of the input filename")
          ;

  boost::program_options::options_description cmdline_options;
  cmdline_options.add(generic).add(config);

  boost::program_options::options_description config_file_options;
  config_file_options.add(config);

  boost::program_options::positional_options_description p;
  p.add("input-file", -1);

  try {
    // parse command line options
    boost::program_options::store(boost::program_options::command_line_parser((int)argc, argv).options(cmdline_options).positional(p).run(), OPT::vm);
    boost::program_options::notify(OPT::vm);
    INIT_WORKING_FOLDER;
    // parse configuration file
    std::ifstream ifs(MAKE_PATH_SAFE(OPT::strConfigFileName));
    if (ifs) {
      boost::program_options::store(parse_config_file(ifs, config_file_options), OPT::vm);
      boost::program_options::notify(OPT::vm);
    }
  }
  catch (const std::exception& e) {
    LOG(e.what());
    return false;
  }

  // initialize the log file
  OPEN_LOGFILE(MAKE_PATH(APPNAME _T("-")+Util::getUniqueName(0)+_T(".log")).c_str());

  // print application details: version and command line
  Util::LogBuild();
  LOG(_T("Command line:%s"), Util::CommandLineToString(argc, argv).c_str());

  // validate input
  Util::ensureValidPath(OPT::strInputFileName);
  Util::ensureUnifySlash(OPT::strInputFileName);
  if (OPT::vm.count("help") || OPT::strInputFileName.IsEmpty()) {
    boost::program_options::options_description visible("Available options");
    visible.add(generic).add(config);
    GET_LOG() << visible;
  }
  if (OPT::strInputFileName.IsEmpty())
    return false;

  // initialize optional options
  Util::ensureValidPath(OPT::strOutputFileName);
  Util::ensureUnifySlash(OPT::strOutputFileName);
  if (OPT::strOutputFileName.IsEmpty())
    OPT::strOutputFileName = Util::getFullFileName(OPT::strInputFileName) + _T(".mvs");

  // initialize global options
  Process::setCurrentProcessPriority((Process::Priority)OPT::nProcessPriority);
  #ifdef _USE_OPENMP
  if (OPT::nMaxThreads != 0)
    omp_set_num_threads(OPT::nMaxThreads);
  #endif

  #ifdef _USE_BREAKPAD
	// start memory dumper
	MiniDumper::Create(APPNAME, WORKING_FOLDER);
	#endif
  return true;
}

// finalize application instance
void Finalize()
{
  #if TD_VERBOSE != TD_VERBOSE_OFF
  // print memory statistics
  Util::LogMemoryInfo();
  #endif

  CLOSE_LOGFILE();
  CLOSE_LOGCONSOLE();
  CLOSE_LOG();
}

int main(int argc, LPCTSTR* argv)
{
  #ifdef _DEBUGINFO
	// set _crtBreakAlloc index to stop in <dbgheap.c> at allocation
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);// | _CRTDBG_CHECK_ALWAYS_DF);
	#endif

  if (!Initialize(argc, argv))
    return EXIT_FAILURE;

  Scene scene(OPT::nMaxThreads);
  // load the base scene
  if (!scene.Load(MAKE_PATH_SAFE(OPT::strInputFileName)))
    return EXIT_FAILURE;
  TD_TIMER_START();
  FOREACH(ID, scene.images) {
    Image& imageData = scene.images[ID];
    boost::regex re(OPT::strRegexExp.c_str());
    if (!boost::regex_match(imageData.name.c_str(), re)) {
      imageData.poseID = NO_ID;
      --scene.nCalibratedImages;
    }
  }

  LOG(_T("%i views remaining"), scene.nCalibratedImages );

  // save the final filtered file
  scene.Save(MAKE_PATH_SAFE(Util::getFullFileName(OPT::strOutputFileName) + _T("_filter.mvs")), (ARCHIVE_TYPE)OPT::nArchiveType);

  Finalize();
  return EXIT_SUCCESS;
}
/*----------------------------------------------------------------*/



