import arim

probes = arim.probes


def test_probes():
    probes.keys()
    repr(probes)
    str(probes)

    assert len(probes) > 0

    for (probe_key, probe) in probes.items():
        assert isinstance(probe, arim.Probe)
        assert probe_key == probe.metadata["short_name"]

    key = tuple(probes.keys())[0]
    probe1 = probes[key]
    probe2 = probes[key]
    assert probe1 is not probe2


def test_probes_makers():
    for (probe_key, maker) in probes._makers.items():
        probe = maker.make()
        assert probe_key == maker.short_name == probe.metadata["short_name"]
        assert maker.long_name == probe.metadata["long_name"]
