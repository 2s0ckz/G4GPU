// G4cout, G4cerr and G4endl.
//
// Geant4 routes its output through a stream so that a UI session can capture it; that is why
// examples write `G4cout << ... << G4endl` rather than std::cout, and why the text shows up in
// a Qt or Xm session's log pane rather than only on the terminal. The same is true here: a
// line written to G4cout goes to stdout and, if a viewer or GUI has installed a sink, into its
// text panel as well.
//
// The sink is a free function rather than a G4UImanager call because G4UImanager includes the
// run manager, which includes almost everything; a stream header that dragged that in would be
// unusable from a header like G4UnitsTable.hh.
#pragma once
#include <cstdio>
#include <functional>
#include <iostream>
#include <ostream>
#include <streambuf>
#include <string>

namespace g4gpu::g4io {

/// Where whole lines go in addition to stdout. Installed by G4UImanager::SetOutputSink.
inline std::function<void(const std::string&)>& Sink() {
  static std::function<void(const std::string&)> s;
  return s;
}

/// Buffers characters until a newline, then writes the line to stdout and to the sink.
///
/// Line-buffered rather than character-forwarding because the sink is a list of lines in a UI
/// panel: forwarding every character would append one entry per character.
class LineBuf : public std::streambuf {
 public:
  explicit LineBuf(std::FILE* out) : out_(out) {}

 protected:
  int overflow(int c) override {
    if (c == traits_type::eof()) { return 0; }
    if (c == '\n') {
      std::fprintf(out_, "%s\n", line_.c_str());
      if (Sink()) { Sink()(line_); }
      line_.clear();
    } else {
      line_.push_back(static_cast<char>(c));
    }
    return c;
  }

  std::streamsize xsputn(const char* s, std::streamsize n) override {
    for (std::streamsize i = 0; i < n; ++i) { overflow(static_cast<unsigned char>(s[i])); }
    return n;
  }

  int sync() override {
    if (!line_.empty()) {
      std::fprintf(out_, "%s", line_.c_str());
      if (Sink()) { Sink()(line_); }
      line_.clear();
    }
    std::fflush(out_);
    return 0;
  }

 private:
  std::FILE* out_;
  std::string line_;
};

inline std::ostream& Cout() {
  static LineBuf buf(stdout);
  static std::ostream s(&buf);
  return s;
}

inline std::ostream& Cerr() {
  static LineBuf buf(stderr);
  static std::ostream s(&buf);
  return s;
}

}  // namespace g4gpu::g4io

// Geant4 spells these as global names. Function-call macros rather than references so that
// the streams are constructed on first use - a global reference initialised at load time
// would depend on the order in which translation units run their initialisers, and the sink
// is installed later anyway.
#define G4cout (g4gpu::g4io::Cout())
#define G4cerr (g4gpu::g4io::Cerr())

/// std::endl on this stream flushes the line to stdout and to the sink.
inline std::ostream& G4endl(std::ostream& os) { return os << '\n'; }
